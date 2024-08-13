package io.github.jbellis;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class Bench {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Config config = new Config();

    public static void main(String[] args) throws IOException {
        config.validateIndexExists();

        // open the index and search for the query
        try (var index = OnDiskGraphIndex.load(new SimpleReaderSupplier());
             var pqvReader = new SimpleReader(config.pqVectorsPath()))
        {
            var pqv = PQVectors.load(pqvReader);
            System.out.println("Loaded index");

            long startTime = System.nanoTime();
            int searches = 1_000;
            var reranked = new AtomicInteger();
            IntStream.range(0, searches).parallel().forEach(__ -> {
                var searcher = new GraphSearcher(index);
                var q = randomVector(pqv);

                var asf = pqv.scoreFunctionFor(q, VectorSimilarityFunction.COSINE);
                var rr = index.getView().rerankerFor(q, VectorSimilarityFunction.COSINE);
                var sf = new SearchScoreProvider(asf, rr);

                var topK = 100;
                var res = searcher.search(sf, topK, Search.rerankK(topK), 0.0f, 0.0f, Bits.ALL);
                reranked.addAndGet(res.getRerankedCount());
            });

            long endTime = System.nanoTime();
            long duration = endTime - startTime;
            System.out.printf("Average search duration: %.2f ms%n", (double) duration / (searches * 1_000_000L));
            System.out.printf("Average reranked count: %.1f%n", (double) reranked.get() / searches);
        }
    }

    // don't use random vectors, they don't share the same distribution as real ones
    static VectorFloat<?> randomVector(PQVectors pqv) {
        var R = ThreadLocalRandom.current();
        VectorFloat<?> v = vts.createFloatVector(pqv.getOriginalSize() / Float.BYTES);
        pqv.getProductQuantization().decode(pqv.get(R.nextInt(pqv.count())), v);
        VectorUtil.l2normalize(v);
        return v;
    }

    static class SimpleReaderSupplier implements ReaderSupplier {
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
