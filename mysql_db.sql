CREATE DATABASE StudentDB;
USE StudentDB;


CREATE TABLE students (
    student_id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    gender VARCHAR(10),
    department VARCHAR(50),
    enrollment_year INT,
    cgpa DECIMAL(3,2)
);


INSERT INTO students (student_id, name, age, gender, department, enrollment_year, cgpa) VALUES
(101, 'Aarav Mehta', 20, 'Male', 'Computer Science', 2023, 8.5),
(102, 'Isha Patel', 21, 'Female', 'Electronics', 2022, 8.9),
(103, 'Rohan Shah', 19, 'Male', 'Mechanical', 2023, 7.8),
(104, 'Sneha Desai', 22, 'Female', 'IT', 2021, 9.1),
(105, 'Yash Trivedi', 20, 'Male', 'Civil', 2022, 7.5),
(106, 'Kavya Joshi', 21, 'Female', 'Computer Science', 2022, 8.7),
(107, 'Devansh Rana', 23, 'Male', 'Electrical', 2020, 7.9),
(108, 'Ritika Bhatt', 20, 'Female', 'IT', 2023, 9.0);

SELECT * FROM students;