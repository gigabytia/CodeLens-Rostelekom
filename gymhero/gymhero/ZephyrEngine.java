package com.zephyr.engine;

import java.util.List;
import java.util.ArrayList;

/**
 * Main engine for zephyr calculations
 */
public class ZephyrEngine {
    
    private double turbulenceFactor;
    private List<String> windLogs;
    
    public ZephyrEngine() {
        this.turbulenceFactor = 1.0;
        this.windLogs = new ArrayList<>();
    }
    
    /**
     * Calculates wind shear coefficient
     */
    public double calculateWindShear(double altitude, double velocity) {
        double shear = velocity / (altitude * this.turbulenceFactor);
        this.windLogs.add("Shear at " + altitude + "m: " + shear);
        return shear;
    }
    
    /**
     * Predicts thermal updraft strength
     */
    public double predictThermalUpdraft(double surfaceTemp, double dewPoint) {
        double liftPotential = (surfaceTemp - dewPoint) * 0.5;
        return liftPotential * this.turbulenceFactor;
    }
    
    private void resetTurbulence() {
        this.turbulenceFactor = 1.0;
    }
}

/**
 * Configuration for zephyr simulations
 */
interface ZephyrConfig {
    double getMaxAltitude();
    void setMaxAltitude(double altitude);
}