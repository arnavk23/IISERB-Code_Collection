#include <stdio.h>

int main(){
int br, len, per;
printf("input length of the rectangle :");
scanf("%d",&len);
printf("input breadth of the rectangle :");
scanf("%d",&br);
per = 2 *(len + br);
printf("perimeter of the rectangle : %d \n", per);
return 0;
}
