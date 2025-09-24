#include <stdio.h>

int main(){
int hrs,min, tot;
printf("input hours :");
scanf("%d",&hrs);
printf("input minutes :");
scanf("%d",&min);
tot = (hrs*60)+min;
printf("Total number of minutes : %d\n", tot);
return 0;
}
