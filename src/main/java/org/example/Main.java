package org.example;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.Feature;
import io.github.jbellis.jvector.graph.disk.FeatureId;
import io.github.jbellis.jvector.graph.disk.InlineVectorValues;
import io.github.jbellis.jvector.graph.disk.InlineVectors;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.util.JsonStringArrayList;

public class Main {
    private static final String DATASET_LOCATION = "E:/datasets/Cohere___wikipedia-2023-11-embed-multilingual-v3/en/0.0.0/37feace541fadccf70579e9f289c3cf8e8b186d7/wikipedia-2023-11-embed-multilingual-v3-train-%s-of-00378.arrow";
    private static final int N_SHARDS = 378;
    private static final int TOTAL_ROWS = 41488110;

    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final ObjectMapper objectMapper = new ObjectMapper();

    private static GraphIndexBuilder builder;
    private static final int DIMENSION = 1024;
    private static OnDiskGraphIndexWriter writer;
    private static ArrayList<ByteSequence<?>> pqVectorsList;
    private static ProductQuantization pq;

    public static void main(String[] args) throws IOException {
        log("Heap space available is %s", Runtime.getRuntime().maxMemory());

        // compute PQ from the first shard
        var pqPath = Paths.get("H:/coherepedia.pq");
        if (pqPath.toFile().exists()) {
            log("Loading PQ from %s", pqPath);
            pq = ProductQuantization.load(new SimpleReader(pqPath));
        } else {
            log("Loading vectors for PQ");
            var vectors = new ArrayList<VectorFloat<?>>();
            forEachRow(filenameFor(0), row -> vectors.add(vts.createFloatVector(row.embedding)));
            log("Computing PQ");
            pq = ProductQuantization.compute(new ListRandomAccessVectorValues(vectors, DIMENSION), DIMENSION * 4 / 64, 256, false);
            pq.write(new DataOutputStream(new BufferedOutputStream(new FileOutputStream(pqPath.toFile()))));
            log("PQ saved to %s", pqPath);
        }

        // set up the index builder
        builder = new GraphIndexBuilder(null,
                                        DIMENSION,
                                        64,
                                        128,
                                        1.5f,
                                        1.2f,
                                        PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
        var indexPath = Paths.get("H:/coherepedia.index");
        var writerBuilder = new OnDiskGraphIndexWriter.Builder(builder.getGraph(), indexPath)
                            .with(new InlineVectors(DIMENSION))
                            .withMapper(new OnDiskGraphIndexWriter.IdentityMapper());
        writer = writerBuilder.build();
        InlineVectorValues inlineVectors = new InlineVectorValues(DIMENSION, writer);
        pqVectorsList = new ArrayList<>(TOTAL_ROWS);
        PQVectors pqVectors = new PQVectors(pq, pqVectorsList);
        builder.setBuildScoreProvider(BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.COSINE, inlineVectors, pqVectors));

        // build the graph
        IntStream.range(0, N_SHARDS).parallel().forEach(Main::processShard);

        log("Running cleanup");
        builder.cleanup();
        writer.write(Map.of());
        writer.close();
        log("Wrote index of %s vectors", builder.getGraph().size());
    }

    @SuppressWarnings("SynchronizeOnNonFinalField")
    private static void processShard(int shardIndex) {
        var n = new LongAdder();
        forEachRow(filenameFor(shardIndex), row -> {
            var vector = vts.createFloatVector(row.embedding);
            int id;
            synchronized (pqVectorsList) {
                id = pqVectorsList.size();
                pqVectorsList.add(pq.encode(vector));
            }
            try {
                writer.writeInline(id, Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(vector)));
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }

            builder.addGraphNode(id, vector);
            n.add(1);
        });
        log("Shard %d: %d rows processed", shardIndex, n.intValue());
    }

    private static String filenameFor(int shardIndex) {
        return String.format(DATASET_LOCATION, String.format("%05d", shardIndex));
    }

    private static void forEachRow(String filename, Consumer<RowData> consumer) {
        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);

        try (FileInputStream fileInputStream = new FileInputStream(Paths.get(filename).toFile());
             var reader = new ArrowStreamReader(fileInputStream, allocator)) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();

            while (reader.loadNextBatch()) {
                for (int i = 0; i < root.getRowCount(); i++) {
                    String id = root.getVector("_id").getObject(i).toString();
                    String title = root.getVector("title").getObject(i).toString();
                    String text = root.getVector("text").getObject(i).toString();
                    var jsonList = (JsonStringArrayList<?>) root.getVector("emb").getObject(i);
                    float[] embedding = convertJsonStringArrayListToFloatArray(jsonList);

                    consumer.accept(new RowData(id, title, text, embedding));
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        allocator.close();
    }

    private static float[] convertJsonStringArrayListToFloatArray(JsonStringArrayList<?> jsonList) {
        float[] floatArray = new float[jsonList.size()];
        for (int i = 0; i < jsonList.size(); i++) {
            try {
                floatArray[i] = objectMapper.readValue(jsonList.get(i).toString(), Float.class);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return floatArray;
    }

    private static void log(String message, Object... args) {
        var timestamp = java.time.LocalTime.now().toString();
        System.out.format(timestamp + ": " + message + "%n", args);
    }

    private record RowData(String id, String title, String text, float[] embedding) {
    }
}

