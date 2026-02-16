package com.namespace.controllers;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.namespace.services.ParticleCommandService;

@RestController
@RequestMapping("/api/events")
@CrossOrigin(origins = "*") // Allow requests from your HTML page
public class EventGeneratorController {

    @Autowired
    private ParticleCommandService commandService;

    /**
     * Endpoint to generate a single particle event
     * POST /api/events/generate
     * Body: { "particleType": "electron", "energy": 1500 }
     */
    @PostMapping("/generate")
    public ResponseEntity<String> generateEvent(@RequestBody EventRequest request) {
        try {
            boolean success = commandService.sendGenerateCommand(
                request.getParticleType(), 
                request.getEnergy()
            );
            
            if (success) {
                return ResponseEntity.ok("{\"status\":\"success\",\"message\":\"Event generation triggered\"}");
            } else {
                return ResponseEntity.status(503)
                    .body("{\"status\":\"error\",\"message\":\"Simulation not connected\"}");
            }
        } catch (Exception e) {
            return ResponseEntity.status(500)
                .body("{\"status\":\"error\",\"message\":\"" + e.getMessage() + "\"}");
        }
    }

    /**
     * Endpoint to generate a batch of events
     * POST /api/events/batch
     * Body: { "particleType": "pion", "energy": 2000, "count": 10 }
     */
    @PostMapping("/batch")
    public ResponseEntity<String> generateBatch(@RequestBody BatchEventRequest request) {
        try {
            boolean success = commandService.sendBatchCommand(
                request.getParticleType(),
                request.getEnergy(),
                request.getCount()
            );
            
            if (success) {
                return ResponseEntity.ok(
                    "{\"status\":\"success\",\"message\":\"Batch generation triggered\",\"count\":" + 
                    request.getCount() + "}"
                );
            } else {
                return ResponseEntity.status(503)
                    .body("{\"status\":\"error\",\"message\":\"Simulation not connected\"}");
            }
        } catch (Exception e) {
            return ResponseEntity.status(500)
                .body("{\"status\":\"error\",\"message\":\"" + e.getMessage() + "\"}");
        }
    }

    /**
     * Check if the simulation is connected and ready
     */
    @GetMapping("/status")
    public ResponseEntity<String> getStatus() {
        boolean connected = commandService.isSimulationConnected();
        return ResponseEntity.ok(
            "{\"connected\":" + connected + 
            ",\"queuedCommands\":" + commandService.getQueuedCommandCount() + "}"
        );
    }

    // DTO Classes
    public static class EventRequest {
        private String particleType; // "electron", "pion", "muon", "gamma"
        private double energy; // in MeV

        public String getParticleType() { return particleType; }
        public void setParticleType(String particleType) { this.particleType = particleType; }
        public double getEnergy() { return energy; }
        public void setEnergy(double energy) { this.energy = energy; }
    }

    public static class BatchEventRequest extends EventRequest {
        private int count;

        public int getCount() { return count; }
        public void setCount(int count) { this.count = count; }
    }
}
