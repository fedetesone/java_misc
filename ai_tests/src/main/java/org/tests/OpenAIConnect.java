package org.tests;

import dev.langchain4j.model.openai.OpenAiChatModel;

public class OpenAIConnect {
    public static void main(String[] args) {
        // Get the API from an environment variable.
        // String apiKey = System.getenv("OPENAI_API_KEY");

        OpenAiChatModel model = OpenAiChatModel.withApiKey("demo");

        String answer = model.generate("How is it going??");

        System.out.println(answer);
    }
}