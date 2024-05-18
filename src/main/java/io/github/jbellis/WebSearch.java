package io.github.jbellis;

import java.io.IOException;

import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import net.openhft.chronicle.map.ChronicleMap;
import net.openhft.chronicle.map.ChronicleMapBuilder;

import static spark.Spark.get;
import static spark.Spark.port;
import static spark.Spark.post;

public class WebSearch {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Config config = new Config();

    private static OnDiskGraphIndex index;
    private static SimpleReader pqvReader;
    private static ChronicleMap<Integer, RowData> contentMap;
    private static PQVectors pqv;
    private static GraphSearcher searcher;
    private static final int PORT = 4567; // Default port for Spark

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
        index = OnDiskGraphIndex.load(new Search.SimpleReaderSupplier());
        pqvReader = new SimpleReader(config.pqVectorsPath());
        contentMap = ChronicleMapBuilder.of((Class<Integer>) (Class) Integer.class, (Class<RowData>) (Class) RowData.class)
                                        .createPersistedTo(config.mapPath().toFile());
        pqv = PQVectors.load(pqvReader);
        searcher = new GraphSearcher(index);
    }

    public static void main(String[] args) {
        port(PORT);
        System.out.format("Listening on port %s%n", PORT);

        get("/", (req, res) -> {
            return "<!DOCTYPE html>" +
                   "<html lang='en'>" +
                   "<head>" +
                   "    <meta charset='UTF-8'>" +
                   "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>" +
                   "    <title>Search</title>" +
                   "    <link href='https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css' rel='stylesheet'>" +
                   "</head>" +
                   "<body>" +
                   "    <div class='container'>" +
                   "        <h1 class='mt-5'>Search</h1>" +
                   "        <form action='/search' method='post'>" +
                   "            <div class='form-group'>" +
                   "                <input type='text' class='form-control' name='query' placeholder='Enter your search query'>" +
                   "            </div>" +
                   "            <button type='submit' class='btn btn-primary'>Search</button>" +
                   "        </form>" +
                   "    </div>" +
                   "</body>" +
                   "</html>";
        });

        post("/search", (req, res) -> {
            String query = req.queryParams("query");
            var q = Search.getVectorEmbedding(query);
            StringBuilder resultsHtml = new StringBuilder("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>Search Results</title><link href='https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css' rel='stylesheet'></head><body><div class='container'><h1 class='mt-5'>Search Results</h1>");

            var asf = pqv.scoreFunctionFor(q, VectorSimilarityFunction.COSINE);
            var rr = index.getView().rerankerFor(q, VectorSimilarityFunction.COSINE);
            var sf = new SearchScoreProvider(asf, rr);

            var topK = 3;
            var results = searcher.search(sf, topK, Search.rerankK(topK), 0.0f, 0.0f, Bits.ALL);
            resultsHtml.append("<ul class='list-group mt-3'>");
            for (var ns : results.getNodes()) {
                var row = contentMap.get(ns.node);
                resultsHtml.append("<li class='list-group-item'>").append(row.toHtml()).append("</li>");
            }
            resultsHtml.append("</ul></div></body></html>");

            return resultsHtml.toString();
        });
    }
}
