
/*
 * Question 1: Smart University Campus Management System
 * Part A: Hash Table with Separate Chaining (Student Database)
 * Part B: Circular Queue (Course Waitlist)
 * Part C: Stack (Library Undo)
 * Part D: Deque with Sentinel Nodes (Cafeteria)
 */

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
using namespace std;

//=================== PART A: HASH TABLE WITH SEPARATE CHAINING ===================

template <typename K, typename V>
class HashNode {
public:
    K key;
    V value;
    HashNode* next;

    HashNode(K k, V v) : key(k), value(v), next(nullptr) {}
};

template <typename K, typename V>
class HashTable {
private:
    HashNode<K, V>** table;
    int capacity;
    int size;
    double loadFactorThreshold;

    // Polynomial accumulation hash function (Lec5)
    int hashFunction(K key) {
        string s = to_string(key);
        long long hash = 0;
        long long power = 1;
        const int PRIME = 31;

        for (int i = 0; i < s.length(); i++) {
            hash = (hash + (s[i] - '0') * power) % capacity;
            power = (power * PRIME) % capacity;
        }
        return (int)((hash % capacity + capacity) % capacity);
    }

    void rehash() {
        int oldCapacity = capacity;
        capacity *= 2;
        HashNode<K, V>** oldTable = table;

        table = new HashNode<K, V>*[capacity];
        for (int i = 0; i < capacity; i++) table[i] = nullptr;
        size = 0;

        for (int i = 0; i < oldCapacity; i++) {
            HashNode<K, V>* curr = oldTable[i];
            while (curr != nullptr) {
                insert(curr->key, curr->value);
                HashNode<K, V>* temp = curr;
                curr = curr->next;
                delete temp;
            }
        }
        delete[] oldTable;
        cout << "[Rehashing completed. New capacity: " << capacity << "]" << endl;
    }

public:
    HashTable(int cap = 16, double lf = 0.75) : capacity(cap), size(0), loadFactorThreshold(lf) {
        table = new HashNode<K, V>*[capacity];
        for (int i = 0; i < capacity; i++) table[i] = nullptr;
    }

    ~HashTable() {
        for (int i = 0; i < capacity; i++) {
            HashNode<K, V>* curr = table[i];
            while (curr != nullptr) {
                HashNode<K, V>* temp = curr;
                curr = curr->next;
                delete temp;
            }
        }
        delete[] table;
    }

    void insert(K key, V value) {
        if ((double)(size + 1) / capacity > loadFactorThreshold) rehash();

        int index = hashFunction(key);
        HashNode<K, V>* newNode = new HashNode<K, V>(key, value);
        newNode->next = table[index];
        table[index] = newNode;
        size++;
    }

    V* search(K key) {
        int index = hashFunction(key);
        HashNode<K, V>* curr = table[index];
        while (curr != nullptr) {
            if (curr->key == key) return &(curr->value);
            curr = curr->next;
        }
        return nullptr;
    }

    bool remove(K key) {
        int index = hashFunction(key);
        HashNode<K, V>* curr = table[index];
        HashNode<K, V>* prev = nullptr;

        while (curr != nullptr) {
            if (curr->key == key) {
                if (prev == nullptr) table[index] = curr->next;
                else prev->next = curr->next;
                delete curr;
                size--;
                return true;
            }
            prev = curr;
            curr = curr->next;
        }
        return false;
    }

    double getLoadFactor() { return (double)size / capacity; }
};

struct StudentInfo {
    string name;
    string department;
    int year;

    StudentInfo() {}
    StudentInfo(string n, string d, int y) : name(n), department(d), year(y) {}

    void display() {
        cout << "Name: " << name << ", Dept: " << department << ", Year: " << year;
    }
};

//=================== PART B: CIRCULAR QUEUE ===================

class CircularQueue {
private:
    int* arr;
    int front;
    int rear;
    int capacity;
    int count;

public:
    CircularQueue(int cap) : capacity(cap), front(0), rear(0), count(0) {
        arr = new int[capacity];
    }

    ~CircularQueue() { delete[] arr; }

    bool isEmpty() { return count == 0; }
    bool isFull() { return count == capacity; }

    void enqueue(int plotID) {
        if (isFull()) throw runtime_error("Queue is full!");
        arr[rear] = plotID;
        rear = (rear + 1) % capacity;
        count++;
        cout << "Added plot " << plotID << " to waitlist. Position: " << count << endl;
    }

    int dequeue() {
        if (isEmpty()) throw runtime_error("Queue is empty!");
        int plotID = arr[front];
        front = (front + 1) % capacity;
        count--;
        return plotID;
    }

    void display() {
        if (isEmpty()) { cout << "Waitlist is empty." << endl; return; }
        cout << "Current waitlist: ";
        for (int i = 0; i < count; i++) {
            cout << arr[(front + i) % capacity];
            if (i < count - 1) cout << " -> ";
        }
        cout << endl;
    }
};

//=================== PART C: STACK (LIBRARY UNDO) ===================

struct CheckoutRecord {
    int bookID;
    int studentID;
    string timestamp;
    CheckoutRecord(int b, int s, string t) : bookID(b), studentID(s), timestamp(t) {}
};

class LibraryStack {
private:
    CheckoutRecord** arr;
    int capacity;
    int top;

    void resize() {
        int newCap = capacity * 2;
        CheckoutRecord** newArr = new CheckoutRecord*[newCap];
        for (int i = 0; i <= top; i++) newArr[i] = arr[i];
        delete[] arr;
        arr = newArr;
        capacity = newCap;
    }

public:
    LibraryStack(int cap = 100) : capacity(cap), top(-1) {
        arr = new CheckoutRecord*[capacity];
    }

    ~LibraryStack() {
        for (int i = 0; i <= top; i++) delete arr[i];
        delete[] arr;
    }

    bool isEmpty() { return top == -1; }

    void push(CheckoutRecord* record) {
        if (top == capacity - 1) resize();
        arr[++top] = record;
        cout << "Book " << record->bookID << " checked out." << endl;
    }

    void undoLastCheckout() {
        if (isEmpty()) { cout << "Nothing to undo!" << endl; return; }
        CheckoutRecord* record = arr[top--];
        cout << "UNDO: Book " << record->bookID << " returned." << endl;
        delete record;
    }

    void displayCurrentBorrowed() {
        if (isEmpty()) { cout << "No books borrowed." << endl; return; }
        cout << "Borrowed books:" << endl;
        for (int i = top; i >= 0; i--) {
            cout << "  Book " << arr[i]->bookID << " by Student " << arr[i]->studentID << endl;
        }
    }
};

//=================== PART D: DEQUE WITH SENTINEL NODES ===================

template <typename T>
class DequeNode {
public:
    T data;
    DequeNode* next;
    DequeNode* prev;
    DequeNode(T d) : data(d), next(nullptr), prev(nullptr) {}
};

template <typename T>
class Deque {
private:
    DequeNode<T>* header;
    DequeNode<T>* trailer;
    int count;

public:
    Deque() : count(0) {
        header = new DequeNode<T>(T());
        trailer = new DequeNode<T>(T());
        header->next = trailer;
        trailer->prev = header;
    }

    ~Deque() {
        while (!isEmpty()) removeFirst();
        delete header;
        delete trailer;
    }

    bool isEmpty() { return count == 0; }

    void addFirst(T elem) {
        DequeNode<T>* newNode = new DequeNode<T>(elem);
        newNode->next = header->next;
        newNode->prev = header;
        header->next->prev = newNode;
        header->next = newNode;
        count++;
    }

    void addLast(T elem) {
        DequeNode<T>* newNode = new DequeNode<T>(elem);
        newNode->next = trailer;
        newNode->prev = trailer->prev;
        trailer->prev->next = newNode;
        trailer->prev = newNode;
        count++;
    }

    T removeFirst() {
        if (isEmpty()) throw runtime_error("Deque is empty!");
        DequeNode<T>* node = header->next;
        T data = node->data;
        header->next = node->next;
        node->next->prev = header;
        delete node;
        count--;
        return data;
    }

    void display() {
        if (isEmpty()) { cout << "No pending orders." << endl; return; }
        cout << "Orders: ";
        DequeNode<T>* curr = header->next;
        while (curr != trailer) {
            cout << curr->data;
            if (curr->next != trailer) cout << " -> ";
            curr = curr->next;
        }
        cout << endl;
    }
};

//=================== MAIN ===================

int main() {
    HashTable<int, StudentInfo> studentDB(16, 0.75);
    CircularQueue waitlist(5);
    LibraryStack library;
    Deque<string> cafeteria;

    int choice;
    while (true) {
        cout << "\n========== SMART CAMPUS MANAGEMENT SYSTEM ==========" << endl;
        cout << "1. Student Database  2. Course Waitlist  3. Library  4. Cafeteria  5. Exit" << endl;
        cout << "Enter choice: ";
        cin >> choice;

        if (choice == 1) {
            cout << "1. Add  2. Search  3. Delete  4. Load Factor" << endl;
            int sub; cin >> sub;
            if (sub == 1) {
                int id, year; string name, dept;
                cout << "ID Name Dept Year: "; cin >> id >> name >> dept >> year;
                studentDB.insert(id, StudentInfo(name, dept, year));
            } else if (sub == 2) {
                int id; cout << "ID: "; cin >> id;
                StudentInfo* s = studentDB.search(id);
                if (s) { s->display(); cout << endl; }
                else cout << "Not found!" << endl;
            } else if (sub == 3) {
                int id; cout << "ID: "; cin >> id;
                cout << (studentDB.remove(id) ? "Removed" : "Not found") << endl;
            } else if (sub == 4) {
                cout << "Load factor: " << studentDB.getLoadFactor() << endl;
            }
        }
        else if (choice == 2) {
            cout << "1. Enqueue  2. Dequeue  3. Display" << endl;
            int sub; cin >> sub;
            try {
                if (sub == 1) {
                    int id; cout << "Plot ID: "; cin >> id;
                    waitlist.enqueue(id);
                } else if (sub == 2) {
                    cout << "Processed: " << waitlist.dequeue() << endl;
                } else if (sub == 3) waitlist.display();
            } catch (runtime_error& e) { cout << "Error: " << e.what() << endl; }
        }
        else if (choice == 3) {
            cout << "1. Checkout  2. Undo  3. Display" << endl;
            int sub; cin >> sub;
            try {
                if (sub == 1) {
                    int b, s; cout << "BookID StudentID: "; cin >> b >> s;
                    library.push(new CheckoutRecord(b, s, "2026-02-06"));
                } else if (sub == 2) library.undoLastCheckout();
                else if (sub == 3) library.displayCurrentBorrowed();
            } catch (runtime_error& e) { cout << "Error: " << e.what() << endl; }
        }
        else if (choice == 4) {
            cout << "1. Regular Order  2. VIP Order  3. Process  4. Display" << endl;
            int sub; cin >> sub;
            try {
                if (sub == 1 || sub == 2) {
                    string order; cout << "Order ID: "; cin >> order;
                    if (sub == 1) { cafeteria.addLast(order + "(R)"); cout << "Regular added." << endl; }
                    else { cafeteria.addFirst(order + "(VIP)"); cout << "VIP added." << endl; }
                } else if (sub == 3) {
                    cout << "Processing: " << cafeteria.removeFirst() << endl;
                } else if (sub == 4) cafeteria.display();
            } catch (runtime_error& e) { cout << "Error: " << e.what() << endl; }
        }
        else if (choice == 5) break;
    }
    return 0;
}
