package io.github.jbellis;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;

import com.cohere.api.Cohere;
import com.cohere.api.requests.EmbedRequest;
import com.cohere.api.types.EmbedInputType;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import net.openhft.chronicle.map.ChronicleMapBuilder;

import static java.lang.Math.max;
import static java.lang.Math.pow;

public class Search {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Config config = new Config();

    public static void main(String[] args) throws IOException {
        config.validateIndexExists();
        config.validateCohereKey();

        // Prompt user for a query string
        System.out.println("Search for: ");
        String query = System.console().readLine();

        // ask Cohere to embed the query
        var q = getVectorEmbedding(query);

        // open the index and search for the query
        try (var index = OnDiskGraphIndex.load(new SimpleReaderSupplier());
             var pqvReader = new SimpleReader(config.pqVectorsPath());
             var contentMap = ChronicleMapBuilder.of((Class<Integer>) (Class) Integer.class, (Class<RowData>) (Class) RowData.class)
                                                 .createPersistedTo(config.mapPath().toFile()))
        {
            var pqv = PQVectors.load(pqvReader);

            var searcher = new GraphSearcher(index);

            var asf = pqv.scoreFunctionFor(q, VectorSimilarityFunction.COSINE);
            var rr = index.getView().rerankerFor(q, VectorSimilarityFunction.COSINE);
            var sf = new SearchScoreProvider(asf, rr);

            var topK = 3;
            var results = searcher.search(sf, topK, rerankK(topK), 0.0f, 0.0f, Bits.ALL);
            System.out.format("%nTop %d results:%n%n", topK);
            for (var ns : results.getNodes()) {
                var row = contentMap.get(ns.node);
                System.out.println(row.prettyPrint());
            }
        }
    }

    private static int rerankK(int topK) {
        var overquery = max(1.0, 0.979 + 4.021 * pow(topK, 0.761)); // f(1) = 5.0, f(100) = 1.1, f(1000) = 1.0
        return (int) (topK * overquery);
    }

    public static VectorFloat<?> getVectorEmbedding(String text) {
        Cohere cohere = Cohere.builder().token(config.getCohereKey()).clientName("snippet").build();
        var request = EmbedRequest.builder().texts(List.of(text)).model("embed-multilingual-v3.0").inputType(EmbedInputType.SEARCH_QUERY).build();
        var response = cohere.embed(request).getEmbeddingsFloats();
        if (response.isEmpty()) {
            throw new IllegalStateException("No embeddings returned -- probably Cohere thinks your text is 'unsafe'");
        }
        return toVector(response.get().getEmbeddings().get(0));
    }

    private static VectorFloat<?> toVector(List<Double> embeddings) {
        var vector = new float[embeddings.size()];
        for (int i = 0; i < embeddings.size(); i++) {
            vector[i] = embeddings.get(i).floatValue();
        }
        return vts.createFloatVector(vector);
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
