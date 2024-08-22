package io.github.jbellis;

import io.github.jbellis.jvector.disk.MemorySegmentReader;
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
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class Bench {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Config config = new Config();

    public static void main(String[] args) throws IOException {
        config.validateIndexExists();

        // open the index and search for the query
        try (var index = OnDiskGraphIndex.load(new MemorySegmentReader.Supplier(config.annPath()));
             var pqvReader = new SimpleReader(config.pqVectorsPath()))
        {
            var pqv = PQVectors.load(pqvReader);
            System.out.println("Loaded index");

            long startTime = System.nanoTime();
            int searches = 100_000;
            var reranked = new AtomicInteger();
            IntStream.range(0, searches).parallel().forEach(__ -> {
                var searcher = new GraphSearcher(index);
                var q = randomVector(pqv.getOriginalSize() / Float.BYTES);

                var asf = pqv.scoreFunctionFor(q, VectorSimilarityFunction.COSINE);
                var view = (OnDiskGraphIndex.View) searcher.getView();
                var rr = view.lvqRerankerFor(q, VectorSimilarityFunction.COSINE);
//                var rr = view.rerankerFor(q, VectorSimilarityFunction.COSINE);
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

    static VectorFloat<?> randomVector(int dim) {
        Random R = ThreadLocalRandom.current();
        VectorFloat<?> vec = vts.createFloatVector(dim);
        for (int i = 0; i < dim; i++) {
            vec.set(i, R.nextFloat());
            if (R.nextBoolean()) {
                vec.set(i, -vec.get(i));
            }
        }
        VectorUtil.l2normalize(vec);
        return vec;
    }
}
