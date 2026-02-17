package com.namespace.services;



import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import java.io.*;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Service
public class ParticleIngestionService {

    private ServerSocket serverSocket;
    private boolean isRunning = true;

    // We use a thread pool to handle multiple Geant4 worker threads
    private final ExecutorService threadPool = Executors.newFixedThreadPool(12);

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    // TODO: Inject your NeuralEngine here later
    // @Autowired
    // private NeuralInferenceService inferenceService;

    @PostConstruct
    public void startServer() {
        // Run the listener in a separate thread so it doesn't block Spring Boot startup
        new Thread(this::listenForParticles).start();
    }

    private void listenForParticles() {
        try {
            int port =5001;
            serverSocket = new ServerSocket(5001, 50, InetAddress.getByName("0.0.0.0"));
            //serverSocket = new ServerSocket(port);
            System.out.println("🚀 LISTENING FOR PARTICLES ON PORT "+port+"...");


            while (isRunning) {
                // Wait for a C++ thread to connect
                Socket clientSocket = serverSocket.accept();
                System.out.println("⚡ New Connection: " + clientSocket.getInetAddress());

                // Hand off to a worker thread
                threadPool.submit(() -> handleConnection(clientSocket));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void handleConnection(Socket socket) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
            String line;
            // Read the stream continuously
            while ((line = reader.readLine()) != null) {
                // LOGIC:
                // 1. Parse CSV (ID, Label, Energies...)
                // 2. Run Inference (Neural Network)
                // 3. Send to Kafka

                // For now, just print raw data to prove it works
                // System.out.println("Received Event: " + line); // Commented out to avoid console spam

                // Push raw event to Kafka "raw-particles" topic
                kafkaTemplate.send("raw-particles", line);
            }
        } catch (IOException e) {
            System.out.println("Connection closed by simulation.");
        }
    }

    @PreDestroy
    public void stop() throws IOException {
        isRunning = false;
        if (serverSocket != null) serverSocket.close();
    }
}