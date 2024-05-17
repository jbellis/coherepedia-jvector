package io.github.jbellis;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import net.openhft.chronicle.map.ChronicleMapBuilder;

public class Search {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Config config = new Config();

    public static void main(String[] args) throws IOException {
        config.validateIndexExists();

        try (var index = OnDiskGraphIndex.load(new SimpleReaderSupplier());
             var pqvReader = new SimpleReader(config.pqVectorsPath());
             var contentMap = ChronicleMapBuilder.of((Class<Integer>) (Class) Integer.class, (Class<RowData>) (Class) RowData.class)
                                                 .createPersistedTo(config.mapPath().toFile()))
        {
            var pqv = PQVectors.load(pqvReader);

            var q = vts.createFloatVector(randomVector(1024));
            var searcher = new GraphSearcher(index);

            var asf = pqv.scoreFunctionFor(q, VectorSimilarityFunction.COSINE);
            var rr = index.getView().rerankerFor(q, VectorSimilarityFunction.COSINE);
            var sf = new SearchScoreProvider(asf, rr);

            var results = searcher.search(sf, 3, Bits.ALL);
            for (var ns : results.getNodes()) {
                var row = contentMap.get(ns.node);
                System.out.println(row.prettyPrint());
            }
        }
    }

    private static float[] randomVector(int dimension) {
        var vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = (float) Math.random() - 0.5f;
        }
        return vector;
    }

    private static class SimpleReaderSupplier implements ReaderSupplier {
        @Override
        public SimpleReader get() {
            try {
                return new SimpleReader(config.annPath());
            }
            catch (FileNotFoundException e) {
                throw new UncheckedIOException(e);
            }
        }
    }
}
