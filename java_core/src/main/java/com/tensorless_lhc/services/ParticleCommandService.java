package com.tensorless_lhc.services;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.stereotype.Service;

import java.io.*;
import java.net.Socket;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import org.springframework.beans.factory.annotation.Value;

@Service
public class ParticleCommandService {

    private Socket commandSocket;
    private PrintWriter commandWriter;
    private boolean isConnected = false;
    private BlockingQueue<String> commandQueue = new LinkedBlockingQueue<>();

    private static final String ELECTRON = "e-";
    private static final String PION = "pi-";
    private static final String MUON = "mu-";
    private static final String GAMMA = "gamma";

    // UPDATED: Port 5003 to match your C++ logs
    private static final int C_PLUS_PLUS_PORT = 5003;

    @Value("${simulation.host:localhost}")
    private String simulationHost;

    @Value("${simulation.port:5003}")
    private int simulationPort;

    @PostConstruct
    public void init() {
        connectToSimulation();
    }


    private synchronized boolean connectToSimulation() {
        // If already connected, check if it's still alive
        if (isConnected && commandSocket != null && !commandSocket.isClosed()) {
            return true;
        }

        try {
            System.out.println("🔄 Connecting to Simulation at " + simulationHost + ":" + simulationPort + "...");            commandSocket = new Socket(simulationHost, simulationPort);
            commandWriter = new PrintWriter(commandSocket.getOutputStream(), true);
            isConnected = true;
            System.out.println("✅ COMMAND CHANNEL ESTABLISHED: Ready to control simulation!");
            return true;
        } catch (IOException e) {
            // Be quiet about failures to avoid spamming logs, just set flag false
            isConnected = false;
        }
        return false;
    }

    public boolean sendGenerateCommand(String particleType, double energy) {
        // Try to reconnect if we lost it
        if (!isConnected) connectToSimulation();

        if (!isConnected) return false;

        String geant4ParticleType = mapParticleType(particleType);
        String command = String.format("GENERATE,%s,%.2f", geant4ParticleType, energy);
        return sendRawCommand(command);
    }

    public boolean sendBatchCommand(String particleType, double energy, int count) {
        if (!isConnected) connectToSimulation();

        if (!isConnected) return false;

        String geant4ParticleType = mapParticleType(particleType);
        String command = String.format("BATCH,%s,%.2f,%d", geant4ParticleType, energy, count);
        return sendRawCommand(command);
    }

    private boolean sendRawCommand(String command) {
        // 1. Check Connection
        if (commandWriter == null || commandWriter.checkError()) {
            System.out.println("⚠️ Connection lost. Reconnecting...");
            boolean reconnected =connectToSimulation();
            if (!reconnected) return false;
        }
        try {
            commandWriter.println(command);
            if (commandWriter.checkError()) {
                throw new IOException("Writer error"); // Detect broken pipe
            }

            commandQueue.add(command);
            System.out.println("📡 Command sent: " + command);
            return true;
        } catch (Exception e) {
            System.err.println("❌ Connection lost during send: " + e.getMessage());
            isConnected = false;
            closeQuietly();
            return false;
        }
    }

    private String mapParticleType(String userType) {
        if (userType == null) return ELECTRON;
        switch (userType.toLowerCase()) {
            case "electron": case "e-": return ELECTRON;
            case "pion": case "pi-": return PION;
            case "muon": case "mu-": return MUON;
            case "gamma": case "photon": return GAMMA;
            default: return ELECTRON;
        }
    }

    public boolean isSimulationConnected() {
        // If we think we aren't connected, try to connect now!
        if (!isConnected) {
            connectToSimulation();
        }
        return isConnected;
    }

    public int getQueuedCommandCount() {
        return commandQueue.size();
    }

    @PreDestroy
    public void cleanup() {
        closeQuietly();
    }

    private void closeQuietly() {
        try {
          //  if (commandWriter != null) commandWriter.close();
            if (commandSocket != null) commandSocket.close();
        } catch (IOException e) {
            // ignore
        }
    }
}