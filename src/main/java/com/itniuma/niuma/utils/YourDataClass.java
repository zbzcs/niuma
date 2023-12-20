package com.itniuma.niuma.utils;

import lombok.Data;

import java.util.List;
@Data
public class YourDataClass {
    private String note;
    private List<Choice> choices;
    private long created;
    private Usage usage;

    // Getter and setter methods

    // Nested classes for inner structure
    @Data
    public static class Choice {
        private String finishReason;
        private Delta delta;

        // Getter and setter methods
    }
    @Data
    public static class Delta {
        private String role;
        private String content;

        // Getter and setter methods
    }
    @Data

    public static class Usage {
        private int promptTokens;
        private int completionTokens;
        private int totalTokens;

        // Getter and setter methods
    }
}
