#include <stdio.h>
#include <stdlib.h>
#include <mysql.h>

int main(void) {
    // Get MySQL credentials from environment variables
    const char* USER = getenv("MYSQL_USER") ? getenv("MYSQL_USER") : "root";
    const char* PASS = getenv("MYSQL_PASS") ? getenv("MYSQL_PASS") : "";
    const char* HOST = "127.0.0.1";
    const char* DB = "studentdb";

    // Initialize MySQL connection
    MYSQL *conn = mysql_init(NULL);
    if (!conn) {
        fprintf(stderr, "mysql_init() failed\n");
        return 1;
    }

    // Connect to MySQL server
    if (!mysql_real_connect(conn, HOST, USER, PASS, NULL, 0, NULL, 0)) {
        fprintf(stderr, "Connection failed: %s\n", mysql_error(conn));
        mysql_close(conn);
        return 1;
    }
    printf("Connected to MySQL successfully!\n");

    // Create database if it doesn't exist
    if (mysql_query(conn, "CREATE DATABASE IF NOT EXISTS studentdb")) {
        fprintf(stderr, "Database creation failed: %s\n", mysql_error(conn));
        mysql_close(conn);
        return 1;
    }
    printf("Database 'studentdb' created/verified.\n");

    // Select the database
    if (mysql_select_db(conn, DB)) {
        fprintf(stderr, "Database selection failed: %s\n", mysql_error(conn));
        mysql_close(conn);
        return 1;
    }

    // Drop table if exists (for fresh start)
    mysql_query(conn, "DROP TABLE IF EXISTS STUDENT");

    // Create STUDENT table
    const char* create_table = 
        "CREATE TABLE STUDENT ("
        "  Roll_No INT PRIMARY KEY,"
        "  Name VARCHAR(100) NOT NULL,"
        "  Dept VARCHAR(20)"
        ")";
    
    if (mysql_query(conn, create_table)) {
        fprintf(stderr, "Table creation failed: %s\n", mysql_error(conn));
        mysql_close(conn);
        return 1;
    }
    printf("Table STUDENT created successfully.\n");

    // Insert sample data
    // Armstrong numbers: 153, 371 (sum of cubes of digits equals the number)
    // Palindrome names: Anna, Nitin, Bob
    const char* insert_data =
        "INSERT INTO STUDENT (Roll_No, Name, Dept) VALUES "
        "(153, 'Armstrong Student', 'CSE'),"
        "(371, 'Another Armstrong', 'ECE'),"
        "(101, 'Anna', 'CSE'),"
        "(102, 'Nitin', 'ECE'),"
        "(103, 'Bob', 'ME'),"
        "(104, 'Alice Smith', 'CSE'),"
        "(105, 'Madam', 'ECE')";

    if (mysql_query(conn, insert_data)) {
        fprintf(stderr, "Data insertion failed: %s\n", mysql_error(conn));
        mysql_close(conn);
        return 1;
    }
    printf("Sample data inserted successfully.\n");
    printf("\nSetup complete! You can now run the Armstrong and Palindrome programs.\n");

    mysql_close(conn);
    return 0;
}

