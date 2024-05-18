package io.github.jbellis;

import java.io.Serializable;

record RowData(String url, String title, String text) implements Serializable {
    public String toMarkdown() {
        return String.format("# %s #%nURL: %s%n%s%n", title, url, text);
    }

    public String toHtml() {
        return "<div class='card mb-3'>" +
               "  <div class='card-body'>" +
               "    <h5 class='card-title'><a href='" + url + "' target='_blank'>" + title + "</a></h5>" +
               "    <p class='card-text'>" + text + "</p>" +
               "  </div>" +
               "</div>";
    }
}
