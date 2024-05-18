package io.github.jbellis;

import static spark.Spark.*;

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
import net.openhft.chronicle.map.ChronicleMap;
import net.openhft.chronicle.map.ChronicleMapBuilder;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;

public class WebSearch {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Config config = new Config();

    private static OnDiskGraphIndex index;
    private static SimpleReader pqvReader;
    private static ChronicleMap<Integer, RowData> contentMap;
    private static PQVectors pqv;
    private static GraphSearcher searcher;
    private static int PORT = 4567;

    static {
        try {
            initializeResources();
        } catch (IOException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private static void initializeResources() throws IOException {
        config.validateIndexExists();
        config.validateCohereKey();

        System.out.println("Loading index");
        index = OnDiskGraphIndex.load(new SimpleReaderSupplier());
        pqvReader = new SimpleReader(config.pqVectorsPath());
        contentMap = ChronicleMapBuilder.of((Class<Integer>) (Class) Integer.class, (Class<RowData>) (Class) RowData.class)
                                        .createPersistedTo(config.mapPath().toFile());
        pqv = PQVectors.load(pqvReader);
        searcher = new GraphSearcher(index);
    }

    public static void main(String[] args) {
        port(PORT); // Default port for Spark
        System.out.format("Listening on port %s%n", PORT);

        get("/", (req, res) -> {
            return "<form action='/search' method='post'>" +
                   "  <input type='text' name='query'>" +
                   "  <input type='submit' value='Search'>" +
                   "</form>";
        });

        post("/search", (req, res) -> {
            String query = req.queryParams("query");
            var q = getVectorEmbedding(query);
            StringBuilder resultsHtml = new StringBuilder("<h1>Search Results</h1>");

            var asf = pqv.scoreFunctionFor(q, VectorSimilarityFunction.COSINE);
            var rr = index.getView().rerankerFor(q, VectorSimilarityFunction.COSINE);
            var sf = new SearchScoreProvider(asf, rr);

            var topK = 3;
            var results = searcher.search(sf, topK, Search.rerankK(topK), 0.0f, 0.0f, Bits.ALL);
            for (var ns : results.getNodes()) {
                var row = contentMap.get(ns.node);
                resultsHtml.append("<p>").append(row.prettyPrint()).append("</p>");
            }

            return resultsHtml.toString();
        });
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
            } catch (FileNotFoundException e) {
                throw new UncheckedIOException(e);
            }
        }
    }
}
