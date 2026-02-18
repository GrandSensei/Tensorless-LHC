package com.tensorless_lhc.controllers;



import com.tensorless_lhc.services.DashboardService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@CrossOrigin // Allow connections from anywhere (useful for testing)
public class DashboardController {

    @Autowired
    private DashboardService dashboardService;

    // The browser will call http://localhost:8080/stream to subscribe
    @GetMapping("/stream")
    public SseEmitter streamEvents() {
        return dashboardService.registerClient();
    }
}