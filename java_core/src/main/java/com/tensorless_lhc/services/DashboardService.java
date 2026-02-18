package com.tensorless_lhc.services;



import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

@Service
public class DashboardService {

    // Thread-safe list to hold all connected browsers (e.g., your laptop + your phone)
    private final List<SseEmitter> emitters = new CopyOnWriteArrayList<>();

    public SseEmitter registerClient() {
        // Create a connection that lasts forever (Long.MAX_VALUE)
        SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);

        // Remove it if the browser closes the tab
        emitter.onCompletion(() -> emitters.remove(emitter));
        emitter.onTimeout(() -> emitters.remove(emitter));

        emitters.add(emitter);
        return emitter;
    }

    public void sendToDashboard(String eventData) {
        // Loop through all connected browsers and send the data
        for (SseEmitter emitter : emitters) {
            try {
                emitter.send(SseEmitter.event().name("particle-event").data(eventData));
            } catch (IOException e) {
                emitters.remove(emitter); // If sending fails, remove the dead connection
            }
        }
    }
}