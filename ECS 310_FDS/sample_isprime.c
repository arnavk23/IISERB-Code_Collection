#include <stdio.h>
#include <stdlib.h>
#include <mysql/mysql.h>

int isPrime(int x){
  if (x < 2) return 0;
  int i = 2;
  int limit = x/2 + 1;
  while (i < limit){
    if (x % i == 0) return 0;
    i += 1;
  }
  return 1;
}

int main(){
  MYSQL *conn;
  MYSQL_RES *res;
  MYSQL_ROW row;
  
  conn = mysql_init(NULL);
  if (!mysql_real_connect(conn, "localhost", "iiserb",
    "password", "Mock", 0, NULL, 0)){
    printf("Connection failed: %s\n", mysql_error(conn));
    return 1;
  }
  
  if (mysql_query(conn, "SELECT Roll_No, Name, CPI FROM Student"))
  {
    printf("Query failed: %s\n", mysql_error(conn));
    return 1;
  }
  
  res = mysql_store_result(conn);
  while ((row = mysql_fetch_row(res))){
    int roll = atoi(row[0]);
    int cpi = atoi(row[2]);
    if (isPrime(roll) && (cpi % 2 == 1)){
      printf("Deleting: Roll_No=%d, Name=%s, CPI=%d\n",
        roll, row[1], cpi);
      char query[200];
      sprintf(query, "DELETE FROM Student WHERE Roll_No=%d", roll);
      mysql_query(conn, query);
    }
  }
  
  mysql_free_result(res);
  mysql_close(conn);
  return 0;
}
