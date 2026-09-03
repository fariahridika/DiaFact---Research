-- DiaFact Database Schema
-- Run in phpMyAdmin or the MySQL CLI.

CREATE DATABASE IF NOT EXISTS diafact
  CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE diafact;

-- Patient registry: each patient gets a permanent numeric ID.
CREATE TABLE IF NOT EXISTS patients (
  id          INT AUTO_INCREMENT PRIMARY KEY,
  name        VARCHAR(150) NOT NULL,
  age         INT NOT NULL,
  gender      VARCHAR(10) NOT NULL,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT chk_patient_age CHECK (age BETWEEN 1 AND 120)
) ENGINE=InnoDB;

-- Each assessment stores inputs and outputs as JSON.
CREATE TABLE IF NOT EXISTS patient_visits (
  id              INT AUTO_INCREMENT PRIMARY KEY,
  patient_id      INT NOT NULL,
  visit_number    INT NOT NULL DEFAULT 1,

  -- Raw clinical values as submitted, after validation.
  input_features  JSON NOT NULL,
  -- The 16 engineered features actually scored.
  model_features  JSON NOT NULL,

  -- Calibrated probability, its label, and the percentage shown to the user.
  risk_score      FLOAT NOT NULL,
  risk_label      VARCHAR(20) NOT NULL,
  risk_percent    FLOAT NOT NULL,

  shap_values     JSON NOT NULL,
  -- { shap_guided: [...], unconstrained: [...], diagnostics: {...} }
  counterfactuals JSON NOT NULL,

  visited_at      DATETIME DEFAULT CURRENT_TIMESTAMP,

  CONSTRAINT fk_visit_patient FOREIGN KEY (patient_id)
    REFERENCES patients(id) ON DELETE CASCADE,

  -- Two concurrent assessments for one patient must not both become "visit 2".
  -- The API allocates the number inside a transaction; this enforces it even
  -- if something else ever writes to the table.
  CONSTRAINT uq_patient_visit UNIQUE (patient_id, visit_number)
) ENGINE=InnoDB;

CREATE INDEX idx_visits_patient ON patient_visits(patient_id, visit_number);
CREATE INDEX idx_patients_created ON patients(created_at);
