import sqlite3

conn = sqlite3.connect("his.db")
cursor = conn.cursor()

# Create patient table
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth DATE NOT NULL,
    gender TEXT CHECK(gender IN ('M', 'F', 'Other')),
    phone TEXT,
    email TEXT,
    address TEXT,
    emergency_contact TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Create visit table
cursor.execute("""
CREATE TABLE IF NOT EXISTS visits (
    visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    visit_date DATE NOT NULL,
    visit_time TIME NOT NULL,
    visit_type TEXT CHECK(visit_type IN ('Emergency', 'Outpatient', 'Inpatient', 'Follow-up')),
    chief_complaint TEXT,
    vital_signs TEXT,
    doctor_id INTEGER,
    status TEXT CHECK(status IN ('Scheduled', 'In Progress', 'Completed', 'Cancelled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
)
""")

# Create diagnosis table
cursor.execute("""
CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id INTEGER NOT NULL,
    icd_code TEXT,
    diagnosis_description TEXT NOT NULL,
    diagnosis_type TEXT CHECK(diagnosis_type IN ('Primary', 'Secondary', 'Differential')),
    severity TEXT CHECK(severity IN ('Mild', 'Moderate', 'Severe')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id)
)
""")

# Commit the changes
conn.commit()

# Insert sample data for all tables - 10 rows each

# Insert sample patients
patients_data = [
    ('John', 'Smith', '1985-03-15', 'M', '555-0101', 'john.smith@email.com', '123 Main St, City, State', 'Jane Smith - 555-0102'),
    ('Sarah', 'Johnson', '1990-07-22', 'F', '555-0103', 'sarah.johnson@email.com', '456 Oak Ave, City, State', 'Mike Johnson - 555-0104'),
    ('Michael', 'Brown', '1978-11-08', 'M', '555-0105', 'michael.brown@email.com', '789 Pine Rd, City, State', 'Lisa Brown - 555-0106'),
    ('Emily', 'Davis', '1995-01-30', 'F', '555-0107', 'emily.davis@email.com', '321 Elm St, City, State', 'Robert Davis - 555-0108'),
    ('David', 'Wilson', '1982-09-12', 'M', '555-0109', 'david.wilson@email.com', '654 Maple Dr, City, State', 'Anna Wilson - 555-0110'),
    ('Lisa', 'Garcia', '1988-05-18', 'F', '555-0111', 'lisa.garcia@email.com', '987 Cedar Ln, City, State', 'Carlos Garcia - 555-0112'),
    ('Robert', 'Martinez', '1975-12-03', 'M', '555-0113', 'robert.martinez@email.com', '147 Birch Ave, City, State', 'Maria Martinez - 555-0114'),
    ('Jennifer', 'Anderson', '1992-08-25', 'F', '555-0115', 'jennifer.anderson@email.com', '258 Spruce St, City, State', 'Tom Anderson - 555-0116'),
    ('William', 'Taylor', '1980-04-14', 'M', '555-0117', 'william.taylor@email.com', '369 Willow Rd, City, State', 'Susan Taylor - 555-0118'),
    ('Amanda', 'Thomas', '1987-10-07', 'F', '555-0119', 'amanda.thomas@email.com', '741 Poplar Dr, City, State', 'James Thomas - 555-0120')
]

cursor.executemany("""
INSERT INTO patients (first_name, last_name, date_of_birth, gender, phone, email, address, emergency_contact)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", patients_data)

# Insert sample visits
visits_data = [
    (1, '2024-01-15', '09:00', 'Outpatient', 'Chest pain', 'BP: 120/80, HR: 75, Temp: 98.6F', 101, 'Completed'),
    (2, '2024-01-16', '10:30', 'Follow-up', 'Diabetes check-up', 'BP: 130/85, HR: 82, Temp: 98.4F', 102, 'Completed'),
    (3, '2024-01-17', '14:15', 'Emergency', 'Severe headache', 'BP: 150/95, HR: 90, Temp: 99.2F', 103, 'Completed'),
    (4, '2024-01-18', '11:00', 'Outpatient', 'Annual physical', 'BP: 115/75, HR: 68, Temp: 98.5F', 101, 'Completed'),
    (5, '2024-01-19', '15:45', 'Inpatient', 'Abdominal pain', 'BP: 125/82, HR: 88, Temp: 100.1F', 104, 'In Progress'),
    (6, '2024-01-20', '08:30', 'Outpatient', 'Skin rash', 'BP: 118/78, HR: 72, Temp: 98.3F', 105, 'Scheduled'),
    (7, '2024-01-21', '13:20', 'Follow-up', 'Post-surgery check', 'BP: 122/79, HR: 76, Temp: 98.7F', 102, 'Scheduled'),
    (8, '2024-01-22', '16:00', 'Emergency', 'Allergic reaction', 'BP: 140/90, HR: 95, Temp: 99.8F', 103, 'Completed'),
    (9, '2024-01-23', '09:45', 'Outpatient', 'Joint pain', 'BP: 128/84, HR: 80, Temp: 98.2F', 104, 'Completed'),
    (10, '2024-01-24', '12:15', 'Follow-up', 'Medication review', 'BP: 135/88, HR: 85, Temp: 98.9F', 101, 'Scheduled')
]

cursor.executemany("""
INSERT INTO visits (patient_id, visit_date, visit_time, visit_type, chief_complaint, vital_signs, doctor_id, status)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", visits_data)

# Insert sample diagnoses
diagnoses_data = [
    (1, 'I25.9', 'Chronic ischemic heart disease, unspecified', 'Primary', 'Moderate', 'Patient advised lifestyle changes and medication'),
    (2, 'E11.9', 'Type 2 diabetes mellitus without complications', 'Primary', 'Mild', 'Blood sugar levels stable, continue current medication'),
    (3, 'G43.909', 'Migraine, unspecified, not intractable, without status migrainosus', 'Primary', 'Severe', 'Prescribed pain medication and rest'),
    (4, 'Z00.00', 'Encounter for general adult medical examination without abnormal findings', 'Primary', 'Mild', 'All vital signs normal, patient healthy'),
    (5, 'K59.00', 'Constipation, unspecified', 'Primary', 'Moderate', 'Recommended dietary changes and increased fluid intake'),
    (6, 'L30.9', 'Dermatitis, unspecified', 'Primary', 'Mild', 'Topical cream prescribed, avoid known allergens'),
    (7, 'Z48.89', 'Encounter for other specified postprocedural aftercare', 'Primary', 'Mild', 'Healing progressing well, no complications'),
    (8, 'T78.40XA', 'Allergy, unspecified, initial encounter', 'Primary', 'Moderate', 'Administered antihistamine, patient stable'),
    (9, 'M25.50', 'Pain in unspecified joint', 'Primary', 'Moderate', 'Anti-inflammatory medication prescribed'),
    (10, 'Z51.81', 'Encounter for therapeutic drug level monitoring', 'Primary', 'Mild', 'Drug levels within therapeutic range')
]

cursor.executemany("""
INSERT INTO diagnoses (visit_id, icd_code, diagnosis_description, diagnosis_type, severity, notes)
VALUES (?, ?, ?, ?, ?, ?)
""", diagnoses_data)

# Commit all changes and close connection
conn.commit()
conn.close()

print("Database seeded successfully with 10 patients, 10 visits, and 10 diagnoses!")
