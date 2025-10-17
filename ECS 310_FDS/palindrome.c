#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <mysql.h>

// Function to check if a string is a palindrome
int is_palindrome(const char* str) {
    if (!str) return 0;
    
    int left = 0;
    int right = strlen(str) - 1;
    
    while (left < right) {
        // Skip spaces and compare letters (case-insensitive)
        while (left < right && str[left] == ' ') left++;
        while (left < right && str[right] == ' ') right--;
        
        // Convert to lowercase and compare
        if (tolower(str[left]) != tolower(str[right])) {
            return 0;  // Not a palindrome
        }
        
        left++;
        right--;
    }
    
    return 1;  // Is a palindrome
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

    // Execute query to get all names
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

    // Check each name for palindrome
    printf("\n=== Palindrome Names Found ===\n\n");
    MYSQL_ROW row;
    int found = 0;
    
    while ((row = mysql_fetch_row(result))) {
        int roll_no = atoi(row[0]);
        const char* name = row[1];
        
        if (is_palindrome(name)) {
            printf("Roll No: %d | Name: %s\n", roll_no, name);
            found++;
        }
    }
    
    if (found == 0) {
        printf("No palindrome names found.\n");
    } else {
        printf("\nTotal palindrome names found: %d\n", found);
    }

    // Cleanup
    mysql_free_result(result);
    mysql_close(conn);
    return 0;
}

