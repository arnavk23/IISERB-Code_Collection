#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mysql.h>

// Function to check if a number is an Armstrong number
int is_armstrong(int num) {
    if (num < 0) return 0;
    
    // Count number of digits
    int original = num;
    int digits = 0;
    int temp = num;
    
    while (temp > 0) {
        digits++;
        temp = temp / 10;
    }
    
    // Calculate sum of each digit raised to power of total digits
    temp = num;
    int sum = 0;
    
    while (temp > 0) {
        int digit = temp % 10;
        sum = sum + (int)pow(digit, digits);
        temp = temp / 10;
    }
    
    // Check if sum equals original number
    return (sum == original);
}

int main(void) {
    // Get MySQL credentials
    const char* USER = getenv("MYSQL_USER") ? getenv("MYSQL_USER") : "root";
    const char* PASS = getenv("MYSQL_PASS") ? getenv("MYSQL_PASS") : "";
    const char* HOST = "127.0.0.1";
    const char* DB = "studentdb";

    // Connect to MySQL
    MYSQL *conn = mysql_init(NULL);
    if (!conn) {
        fprintf(stderr, "mysql_init() failed\n");
        return 1;
    }

    if (!mysql_real_connect(conn, HOST, USER, PASS, DB, 0, NULL, 0)) {
        fprintf(stderr, "Connection failed: %s\n", mysql_error(conn));
        mysql_close(conn);
        return 1;
    }

    // Execute query to get all roll numbers
    if (mysql_query(conn, "SELECT Roll_No, Name FROM STUDENT")) {
        fprintf(stderr, "Query failed: %s\n", mysql_error(conn));
        mysql_close(conn);
        return 1;
    }

    // Get the result
    MYSQL_RES *result = mysql_store_result(conn);
    if (!result) {
        fprintf(stderr, "Failed to get result: %s\n", mysql_error(conn));
        mysql_close(conn);
        return 1;
    }

    // Check each roll number
    printf("\n=== Armstrong Numbers Found ===\n\n");
    MYSQL_ROW row;
    int found = 0;
    
    while ((row = mysql_fetch_row(result))) {
        int roll_no = atoi(row[0]);
        const char* name = row[1];
        
        if (is_armstrong(roll_no)) {
            printf("Roll No: %d | Name: %s\n", roll_no, name);
            found++;
        }
    }
    
    if (found == 0) {
        printf("No Armstrong numbers found.\n");
    } else {
        printf("\nTotal Armstrong numbers found: %d\n", found);
    }

    // Cleanup
    mysql_free_result(result);
    mysql_close(conn);
    return 0;
}

