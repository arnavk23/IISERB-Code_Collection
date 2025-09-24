// C Program to demonstrate the use of array
#include <stdio.h>

int main()
{
int arr[5] = { 10, 20, 30, 40, 50 }; // array declaration and initialization
arr[3] = 100; // modifying element at index 2
printf("Elements in Array: "); // traversing array using for loop
for (int i = 0; i < 5; i++)
{
printf("%d ", arr[i]);
}
return 0;
}
