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

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static void initializeResources() throws IOException {
        config.validateIndexExists();
        config.validateCohereKey();

        System.out.printf("Loading index from %s%n", config.annPath());
        index = OnDiskGraphIndex.load(new Search.SimpleReaderSupplier());
        try (var pqvReader = new SimpleReader(config.pqVectorsPath())) {
            pqv = PQVectors.load(pqvReader);
        }
        contentMap = ChronicleMapBuilder.of((Class<Integer>) (Class) Integer.class, (Class<RowData>) (Class) RowData.class)
                                        .createPersistedTo(config.mapPath().toFile());
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
                    "    <footer class='mt-5 bg-dark text-white py-3'>" +
                    "        <div class='container'>" +
                    "            <p class='mb-0'><a href='https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3' class='text-white' style='text-decoration: underline;'>Cohere V3 Wikipedia dataset</a> built with <a href='https://github.com/jbellis/jvector' class='text-white' style='text-decoration: underline;'>JVector</a></p>" +
                    "        </div>" +
                    "    </footer>" +
                   "</body>" +
                   "</html>";
        });

        post("/search", (req, res) -> {
            // ask Cohere to turn the search string into a vector embedding
            String query = req.queryParams("query");
            var q = Search.getVectorEmbedding(query);

            // approximate score function for the first pass
            var asf = pqv.scoreFunctionFor(q, VectorSimilarityFunction.COSINE);
            // reranking function for the second pass
            var rr = index.getView().rerankerFor(q, VectorSimilarityFunction.COSINE);
            // bundle them together
            var sf = new SearchScoreProvider(asf, rr);

            // perform the search
            var topK = 5;
            long start = System.nanoTime();
            var results = searcher.search(sf, // score function
                                          topK, // this many final results
                                          Search.rerankK(topK), // out of this many approximate results
                                          0.0f, // minimum similarity threshold, out of scope for this example
                                          0.0f, // rerankFloor, out of scope for this example
                                          Bits.ALL); // IDs to allow in the results
            System.out.format("Search took %,d ms%n", (System.nanoTime() - start) / 1_000_000L);

            // render the results
            StringBuilder resultsHtml = new StringBuilder("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>Search Results</title><link href='https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css' rel='stylesheet'></head><body><div class='container'><h1 class='mt-5'>Search Results for \"").append(query).append("\"</h1>");
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
