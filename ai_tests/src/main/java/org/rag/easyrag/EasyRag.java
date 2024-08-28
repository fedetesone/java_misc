package org.rag.easyrag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;


import java.util.List;
import java.util.Objects;

public class EasyRag {

    // private static final String API_KEY = System.getenv("OPENAI_API_KEY");
    private static final String API_KEY = "demo";

    public static void main(String[] args) {
        String url = Objects.requireNonNull(EasyRag.class.getClassLoader().getResource("easyrag_documents")).getPath();
        List<Document> documents = FileSystemDocumentLoader.loadDocuments(url);

        // Creating in-memory embedding store.
        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        EmbeddingStoreIngestor.ingest(documents, embeddingStore);

        // Creating assistant.
        Assistant assistant = buildAssistant(embeddingStore);

        String answer = assistant.chat("What's the best restaurant the US?");

        System.out.println(answer);
        // Example answer:
        // Based on the information provided, the best restaurant in the US is "La Bella Italia" located at 123 5th Avenue, New York, NY 10001.
        // Known for its authentic Italian cuisine, some popular menu items include Margherita Pizza, Fettuccine Alfredo, Bruschetta, Tiramisu,
        // and Panna Cotta. Reviews from customers like Alice, Cathy, and David praise the amazing food, best Italian restaurant in NYC,
        // and cozy atmosphere. While Evelyn mentions the prices being a bit high for portion sizes, overall, La Bella Italia stands out
        // as a top choice for Italian dining in the US.
    }

    private static Assistant buildAssistant(InMemoryEmbeddingStore<TextSegment> embeddingStore) {
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2) // on each interaction we will retrieve the 2 most relevant segments
                .minScore(0.5) // Minimum similarity allowed for the context sent to the LLM.
                .build();

        return AiServices.builder(Assistant.class)
                .chatLanguageModel(OpenAiChatModel.withApiKey(API_KEY))
                .chatMemory(MessageWindowChatMemory.withMaxMessages(1))
                .contentRetriever(contentRetriever)
                .build();
    }
}
