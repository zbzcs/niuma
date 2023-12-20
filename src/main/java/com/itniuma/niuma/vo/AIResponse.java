package com.itniuma.niuma.vo;

import lombok.Data;

import java.util.List;

@Data
public class AIResponse {
    private String note;
    private List<Choice> choices;
    private long created;
    private String id;
    private Usage usage;


    public static class Choice {
        private String finishReason;
        private Delta delta;


        public static class Delta {
            private String role;
            private String content;

        }
    }

    public static class Usage {
        private int promptTokens;
        private int completionTokens;
        private int totalTokens;

    }
}
