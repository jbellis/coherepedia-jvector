package io.github.jbellis;

import java.io.Serializable;

record RowData(String url, String title, String text) implements Serializable {
    public String prettyPrint() {
        return String.format("# %s #%nURL: %s%n%s%n", title, url, text);
    }
}
