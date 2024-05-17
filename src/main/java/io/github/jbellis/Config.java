package io.github.jbellis;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

public class Config {
    private final Path datasetPath;
    private final Path indexPath;
    private final int divisor;

    public Config() {
        Properties props = new Properties();
        try (FileInputStream fis = new FileInputStream("config.properties")) {
            props.load(fis);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load configuration properties.", e);
        }
        datasetPath = Path.of(props.getProperty("dataset_location")).resolve("Cohere___wikipedia-2023-11-embed-multilingual-v3/en/0.0.0/37feace541fadccf70579e9f289c3cf8e8b186d7/wikipedia-2023-11-embed-multilingual-v3-train-%s-of-00378.arrow");
        indexPath = Path.of(props.getProperty("index_location"));
        divisor = Integer.parseInt(props.getProperty("divisor"));
    }

    public void validateDatasetPath() {
        var samplePath = Path.of(filenameForShard(0));
        if (!Files.exists(samplePath)) {
            System.out.format("Dataset does not exist at %s%nThis probably means you need to run download.py first", samplePath);
            System.exit(1);
        }
    }

    public String filenameForShard(int shardIndex) {
        return String.format(datasetPath.toString(), String.format("%05d", shardIndex));
    }

    public void maybeCreateIndexDirectory() {
        if (!Files.exists(indexPath)) {
            try {
                Files.createDirectory(indexPath);
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    public Path annPath() {
        return indexPath.resolve("coherepedia.ann");
    }

    public Path mapPath() {
        return indexPath.resolve("coherepedia.map");
    }

    public int getDivisor() {
        return divisor;
    }

    public Path pqPath() {
        return indexPath.resolve("coherepedia.pq");
    }

    public Path lvqPath() {
        return indexPath.resolve("coherepedia.lvq");
    }
}
