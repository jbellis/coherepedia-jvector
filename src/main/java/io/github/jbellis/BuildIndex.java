package io.github.jbellis;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.Feature;
import io.github.jbellis.jvector.graph.disk.FeatureId;
import io.github.jbellis.jvector.graph.disk.LVQ;
import io.github.jbellis.jvector.graph.disk.LvqVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import net.openhft.chronicle.map.ChronicleMap;
import net.openhft.chronicle.map.ChronicleMapBuilder;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.util.JsonStringArrayList;

public class BuildIndex {
    private static final Config config = new Config();
    private static final int N_SHARDS = 378;
    private static final int TOTAL_ROWS = 41488110 / config.getDivisor();

    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static GraphIndexBuilder builder;
    private static final int DIMENSION = 1024;
    private static OnDiskGraphIndexWriter writer;
    private static ArrayList<ByteSequence<?>> pqVectorsList;
    private static ProductQuantization pq;
    private static LocallyAdaptiveVectorQuantization lvq;
    private static ChronicleMap<Integer, RowData> contentMap;

    @SuppressWarnings({"unchecked", "rawtypes"})
    public static void main(String[] args) throws IOException {
        log("Heap space available is %s", Runtime.getRuntime().maxMemory());

        // setup
        config.validateDatasetPath();
        config.maybeCreateIndexDirectory();
        var indexPath = config.annPath();
        var mapPath = config.mapPath();
        if (Files.exists(indexPath) || Files.exists(mapPath)) {
            log("Index already exists at %s + %s -- remove these manually to rebuild", indexPath, mapPath);
            System.exit(1);
        }

        // compute PQ from the first shard
        var pqPath = config.pqPath();
        var lvqPath = config.lvqPath();
        if (pqPath.toFile().exists() && lvqPath.toFile().exists()) {
            log("Loading PQ and LVQ from previously saved files");
            pq = ProductQuantization.load(new SimpleReader(pqPath));
            lvq = LocallyAdaptiveVectorQuantization.load(new SimpleReader(lvqPath));
        } else {
            log("Loading vectors for quantization");
            var vectors = new ArrayList<VectorFloat<?>>();
            forEachRow(config.filenameForShard(0), (row, embedding) -> vectors.add(vts.createFloatVector(embedding)));
            var ravv = new ListRandomAccessVectorValues(vectors, DIMENSION);
            log("Computing PQ");
            pq = ProductQuantization.compute(ravv, DIMENSION * 4 / 64, 256, false);
            try (var out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(pqPath.toFile())))) {
                pq.write(out);
            }
            log("Computing LVQ");
            lvq = LocallyAdaptiveVectorQuantization.compute(ravv);
            try (var out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(lvqPath.toFile())))) {
                lvq.write(out);
            }
            log("Quantization complete");
        }

        // set up the index builder
        builder = new GraphIndexBuilder(null, // we'll set the score provider later
                                        DIMENSION,
                                        48,   // degree
                                        128,  // search width during construction
                                        1.2f, // allow exceeding degree by this much temporarily
                                        1.2f, // alpha diversity parameter
                                        PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
        var lvqFeature = new LVQ(lvq);
        var writerBuilder = new OnDiskGraphIndexWriter.Builder(builder.getGraph(), indexPath)
                            .with(lvqFeature)
                            .withMapper(new OnDiskGraphIndexWriter.IdentityMapper());
        writer = writerBuilder.build();
        var inlineVectors = new LvqVectorValues(DIMENSION, lvqFeature, writer);
        pqVectorsList = new ArrayList<>(TOTAL_ROWS);
        PQVectors pqVectors = new PQVectors(pq, pqVectorsList);
        builder.setBuildScoreProvider(BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.COSINE, inlineVectors, pqVectors));

        // set up Chronicle Map
        log("Creating index for %,d rows", TOTAL_ROWS);
        contentMap = ChronicleMapBuilder.of((Class<Integer>) (Class) Integer.class, (Class<RowData>) (Class) RowData.class)
                                        .averageValueSize(768) // url (~200) + title (~50) + text (~500)
                                        .entries(TOTAL_ROWS)
                                        .createPersistedTo(mapPath.toFile());

        // build the graph
        IntStream.range(0, N_SHARDS).parallel().forEach(BuildIndex::processShard);
        // sanity checks
        if (pqVectorsList.size() != builder.getGraph().size()) {
            throw new IllegalStateException("PQ vectors and graph size mismatch");
        }
        if (contentMap.size() != builder.getGraph().size()) {
            throw new IllegalStateException("Content map and graph size mismatch");
        }

        log("Final cleanup");
        builder.cleanup();

        log("Writing edge lists");
        writer.write(Map.of());

        writer.close();
        contentMap.close();

        log("Writing compressed vectors");
        try (var pqvOut = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(config.pqVectorsPath().toFile())))) {
            pqVectors.write(pqvOut);
        }

        log("Wrote index of %s vectors", builder.getGraph().size());
    }

    @SuppressWarnings("SynchronizeOnNonFinalField")
    private static void processShard(int shardIndex) {
        forEachRow(config.filenameForShard(shardIndex), (row, embedding) -> {
            var vector = vts.createFloatVector(embedding);
            // wrap raw embedding in VectorFloat
            // id is derived from inserting into the PQ list
            int id;
            synchronized (pqVectorsList) {
                id = pqVectorsList.size();
                pqVectorsList.add(pq.encode(vector));
            }
            if (id % 100_000 == 0) {
                log("%,d rows processed", id);
            }

            // write the vector to the index so it can be read by rerank (call is threadsafe)
            try {
                writer.writeInline(id, Feature.singleState(FeatureId.LVQ, new LVQ.State(lvq.encode(vector))));
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            // add the vector to the graph (also threadsafe)
            builder.addGraphNode(id, vector);
            contentMap.put(id, row);
        });
        log("Shard %d completed", shardIndex);
    }

    private static void forEachRow(String filename, BiConsumer<RowData, float[]> consumer) {
        try (var allocator = new RootAllocator();
             var fileInputStream = new FileInputStream(filename);
             var reader = new ArrowStreamReader(fileInputStream, allocator))
        {
            var root = reader.getVectorSchemaRoot();

            while (reader.loadNextBatch()) {
                for (int i = 0; i < root.getRowCount() / config.getDivisor(); i++) {
                    String url = root.getVector("url").getObject(i).toString();
                    String title = root.getVector("title").getObject(i).toString();
                    String text = root.getVector("text").getObject(i).toString();
                    // TODO is there a way to read floats directly instead of going through String first?
                    var jsonList = (JsonStringArrayList<?>) root.getVector("emb").getObject(i);
                    float[] embedding = convertToFloatArray(jsonList);

                    consumer.accept(new RowData(url, title, text), embedding);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static float[] convertToFloatArray(JsonStringArrayList<?> jsonList) {
        float[] floatArray = new float[jsonList.size()];
        for (int i = 0; i < jsonList.size(); i++) {
            floatArray[i] = Float.parseFloat(jsonList.get(i).toString());
        }
        return floatArray;
    }

    private static void log(String message, Object... args) {
        var timestamp = LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss"));
        System.out.format(timestamp + ": " + message + "%n", args);
    }
}
