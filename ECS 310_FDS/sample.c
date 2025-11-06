#include <stdio.h>
#include <mysql/mysql.h>
#include <math.h>

int isArmstrong(int n){
  int temp = n, sum = 0, digits = 0;
  while (temp){
    digits++;
    temp /= 10; // Division by 10
  }
  temp = n;
  while (temp){
    int d = temp % 10; // Modulo by 10 i.e., remainder
    sum += pow(d, digits); // power of the modulo
    temp /= 10; // Getting the next digit
  }
  if (sum==n) // ‘n’ is Armstrong number, if equal to sum
    return(1);
  return(0);
}

int main(){
  MYSQL *conn; // Establishing MySQL connection
  MYSQL_RES *res;
  MYSQL_ROW row;
  
  conn = mysql_init(NULL);
  if (!mysql_real_connect(conn, "localhost", "iiserb",
    "password", "Mock", 0, NULL, 0)){
    printf("Connection failed: %s\n", mysql_error(conn));
    return 1;
  }
  
  if (mysql_query(conn, "SELECT Roll_No, Name, CPI FROM
    Student"))
  {
    printf("Query failed: %s\n", mysql_error(conn));
    return 1;
  }
  
  res = mysql_store_result(conn);
  while ((row = mysql_fetch_row(res))){
    int roll = atoi(row[0]); // ASCII Code to Integer
    int cpi = atoi(row[2]);
    if (isArmstrong(roll) && (cpi % 2 == 1)){
      printf("Deleting: Roll_No=%d, Name=%s, CPI=%d\n",
        roll, row[1], cpi);
      char query[200];
      sprintf(query, "DELETE FROM Student WHERE 
        Roll_No=%d", roll);
      mysql_query(conn, query);
    }
  }
  
  mysql_free_result(res);
  mysql_close(conn);
  return 0;
}
