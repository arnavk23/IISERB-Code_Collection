#include <stdio.h>

int main(){
int half,quat,dime,nick,pen;
float dollar;
printf("input number of half-pennies :");
scanf("%d",&half);
printf("input number of quarters :");
scanf("%d",&quat);
printf("input number of dimes :");
scanf("%d",&dime);
printf("input number of nickels :");
scanf("%d",&nick);
printf("input number of pennies :");
scanf("%d",&pen);
dollar = (0.5 * half) + (0.25 * quat) + (0.10 * dime) +(0.05 * nick) + (0.01 * pen);
printf("Total dollars : %f\n", dollar);
return 0;
}
