
/*
 * Question 3: Real-Time Traffic Management System
 * Part A: Hash Table with Universal Hashing + MAD (Vehicle Registry)
 * Part B: Priority Queue using Vector (Traffic Signals)
 * Part C: Two Stacks for Navigation (Route History)
 * Part D: Doubly Linked List with Position ADT (Accident Tracking)
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
using namespace std;

//=================== PART A: UNIVERSAL HASHING WITH MAD ===================

struct VehicleInfo {
    string owner;
    string type;
    string regDate;

    VehicleInfo() {}
    VehicleInfo(string o, string t, string d) : owner(o), type(t), regDate(d) {}
};

// Sorted linked list node for chaining
struct VehicleNode {
    string plate;
    VehicleInfo info;
    VehicleNode* next;

    VehicleNode(string p, VehicleInfo i) : plate(p), info(i), next(nullptr) {}
};

class VehicleRegistry {
private:
    VehicleNode** table;
    int capacity;
    int size;
    int a, b;  // MAD parameters
    const double A = 0.6180339887;  // Golden ratio conjugate (Fibonacci hashing)

    // Polynomial accumulation for string keys
    long long hashCode(string key) {
        long long hash = 0;
        long long power = 1;
        const int PRIME = 31;

        for (char c : key) {
            hash = (hash + c * power);
            power *= PRIME;
        }
        return hash;
    }

    // MAD compression: |ak + b| mod m
    int madCompression(long long k) {
        return abs((a * k + b) % capacity);
    }

    // Multiplication method: floor(m * (kA mod 1))
    int multiplicationHash(long long k) {
        double frac = k * A - floor(k * A);
        return (int)(capacity * frac);
    }

    // Universal hash function selection
    int hashFunction(string key, bool useMAD = true) {
        long long k = hashCode(key);
        if (useMAD) return madCompression(k);
        else return multiplicationHash(k);
    }

    void rehash() {
        int oldCap = capacity;
        capacity *= 2;
        VehicleNode** oldTable = table;

        // New MAD parameters
        a = rand() % (capacity - 1) + 1;
        b = rand() % capacity;

        table = new VehicleNode*[capacity];
        for (int i = 0; i < capacity; i++) table[i] = nullptr;
        size = 0;

        // Reinsert
        for (int i = 0; i < oldCap; i++) {
            VehicleNode* curr = oldTable[i];
            while (curr != nullptr) {
                registerVehicle(curr->plate, curr->info);
                VehicleNode* temp = curr;
                curr = curr->next;
                delete temp;
            }
        }
        delete[] oldTable;
        cout << "[Rehashed. New capacity: " << capacity << "]" << endl;
    }

public:
    VehicleRegistry(int cap = 16) : capacity(cap), size(0) {
        srand(time(nullptr));
        a = rand() % (capacity - 1) + 1;
        b = rand() % capacity;
        table = new VehicleNode*[capacity];
        for (int i = 0; i < capacity; i++) table[i] = nullptr;
    }

    void registerVehicle(string plate, VehicleInfo info) {
        if ((double)(size + 1) / capacity > 0.7) rehash();

        int idx = hashFunction(plate);
        VehicleNode* newNode = new VehicleNode(plate, info);

        // Insert in sorted order (by plate number)
        if (table[idx] == nullptr || table[idx]->plate > plate) {
            newNode->next = table[idx];
            table[idx] = newNode;
        } else {
            VehicleNode* curr = table[idx];
            while (curr->next != nullptr && curr->next->plate < plate) 
                curr = curr->next;
            newNode->next = curr->next;
            curr->next = newNode;
        }
        size++;
    }

    VehicleInfo* searchVehicle(string plate) {
        int idx = hashFunction(plate);
        VehicleNode* curr = table[idx];
        while (curr != nullptr) {
            if (curr->plate == plate) return &(curr->info);
            if (curr->plate > plate) break;  // Sorted, so not found
            curr = curr->next;
        }
        return nullptr;
    }

    void displayLoadFactor() {
        cout << "Load factor: " << (double)size / capacity << " (Threshold: 0.7)" << endl;
    }
};

//=================== PART B: PRIORITY QUEUE FOR TRAFFIC ===================

enum VehicleType { NORMAL, EMERGENCY };

struct TrafficVehicle {
    string vehicleID;
    VehicleType type;
    int arrivalTime;
    int priority;  // Lower = higher priority

    TrafficVehicle(string id, VehicleType t, int at) : vehicleID(id), type(t), arrivalTime(at) {
        priority = (type == EMERGENCY) ? 0 : arrivalTime;
    }
};

class TrafficPriorityQueue {
private:
    vector<TrafficVehicle> heap;  // Min-heap based on priority
    string direction;

    void heapifyUp(int idx) {
        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (heap[parent].priority > heap[idx].priority) {
                swap(heap[parent], heap[idx]);
                idx = parent;
            } else break;
        }
    }

    void heapifyDown(int idx) {
        int n = heap.size();
        while (true) {
            int smallest = idx;
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;

            if (left < n && heap[left].priority < heap[smallest].priority) smallest = left;
            if (right < n && heap[right].priority < heap[smallest].priority) smallest = right;

            if (smallest != idx) {
                swap(heap[idx], heap[smallest]);
                idx = smallest;
            } else break;
        }
    }

public:
    TrafficPriorityQueue(string dir) : direction(dir) {}

    void enqueue(string id, VehicleType type, int arrival) {
        TrafficVehicle v(id, type, arrival);
        heap.push_back(v);
        heapifyUp(heap.size() - 1);

        if (type == EMERGENCY) 
            cout << "EMERGENCY VEHICLE " << id << " added to " << direction << "! Priority: 0" << endl;
    }

    TrafficVehicle dequeue() {
        if (heap.empty()) throw runtime_error("Queue empty!");

        TrafficVehicle v = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        if (!heap.empty()) heapifyDown(0);

        cout << "Processing from " << direction << ": " << v.vehicleID;
        if (v.type == EMERGENCY) cout << " [EMERGENCY]";
        cout << endl;
        return v;
    }

    bool isEmpty() { return heap.empty(); }

    void display() {
        cout << direction << " queue: ";
        for (auto& v : heap) {
            cout << v.vehicleID << "(P" << v.priority << ") ";
        }
        cout << endl;
    }
};

class FourWayIntersection {
private:
    TrafficPriorityQueue north;
    TrafficPriorityQueue south;
    TrafficPriorityQueue east;
    TrafficPriorityQueue west;

public:
    FourWayIntersection() : north("NORTH"), south("SOUTH"), east("EAST"), west("WEST") {}

    void addVehicle(string dir, string id, VehicleType type, int arrival) {
        if (dir == "NORTH") north.enqueue(id, type, arrival);
        else if (dir == "SOUTH") south.enqueue(id, type, arrival);
        else if (dir == "EAST") east.enqueue(id, type, arrival);
        else if (dir == "WEST") west.enqueue(id, type, arrival);
    }

    void processAllDirections() {
        cout << "\nProcessing all directions..." << endl;
        if (!north.isEmpty()) north.dequeue();
        if (!south.isEmpty()) south.dequeue();
        if (!east.isEmpty()) east.dequeue();
        if (!west.isEmpty()) west.dequeue();
    }

    void displayAll() {
        north.display(); south.display(); east.display(); west.display();
    }
};

//=================== PART C: TWO STACKS FOR NAVIGATION ===================

template <typename T>
class NavigationSystem {
private:
    vector<T> forwardStack;  // Current route
    vector<T> backwardStack; // Backtracking history

public:
    void goForward(T location) {
        forwardStack.push_back(location);
        // Clear backward stack when new forward move
        backwardStack.clear();
        cout << "Moved forward to: " << location << endl;
    }

    void goBack() {
        if (forwardStack.empty()) {
            cout << "Cannot go back! Already at start." << endl;
            return;
        }
        T loc = forwardStack.back();
        forwardStack.pop_back();
        backwardStack.push_back(loc);
        cout << "Went back from: " << loc << endl;
    }

    void goForwardAfterBack(T location) {
        // Clear backward history and go new direction
        backwardStack.clear();
        goForward(location);
    }

    void showCurrentRoute() {
        cout << "\n=== Current Route ===" << endl;
        cout << "Start -> ";
        for (auto& loc : forwardStack) cout << loc << " -> ";
        cout << "CURRENT POSITION" << endl;

        if (!backwardStack.empty()) {
            cout << "Back history: ";
            for (auto it = backwardStack.rbegin(); it != backwardStack.rend(); ++it)
                cout << *it << " ";
            cout << endl;
        }
    }

    void undoLastMove() {
        if (!backwardStack.empty()) {
            T loc = backwardStack.back();
            backwardStack.pop_back();
            forwardStack.push_back(loc);
            cout << "Redid: " << loc << endl;
        } else cout << "Nothing to redo." << endl;
    }
};

//=================== PART D: DOUBLY LINKED LIST WITH POSITION ADT ===================

// Forward declaration
class Position;
class DoublyLinkedList;

// Node structure
class DListNode {
public:
    string data;  // Simplified for Accident string representation
    DListNode* prev;
    DListNode* next;

    DListNode(string d) : data(d), prev(nullptr), next(nullptr) {}
};

// Position ADT
class Position {
private:
    DListNode* node;

public:
    Position(DListNode* n = nullptr) : node(n) {}

    string& element() { return node->data; }
    bool isNull() { return node == nullptr; }
    DListNode* getNode() { return node; }

    friend class DoublyLinkedList;
};

// Doubly Linked List
class DoublyLinkedList {
private:
    DListNode* header;
    DListNode* trailer;
    int count;

public:
    DoublyLinkedList() : count(0) {
        header = new DListNode("");
        trailer = new DListNode("");
        header->next = trailer;
        trailer->prev = header;
    }

    ~DoublyLinkedList() {
        clear();
        delete header;
        delete trailer;
    }

    void clear() {
        DListNode* curr = header->next;
        while (curr != trailer) {
            DListNode* temp = curr;
            curr = curr->next;
            delete temp;
        }
        header->next = trailer;
        trailer->prev = header;
        count = 0;
    }

    // Insert at position (before node at position p)
    void insertAt(Position p, string elem) {
        DListNode* node = p.getNode();
        DListNode* newNode = new DListNode(elem);

        newNode->next = node;
        newNode->prev = node->prev;
        node->prev->next = newNode;
        node->prev = newNode;
        count++;
    }

    Position first() { return Position(header->next); }
    Position last() { return Position(trailer); }

    bool isFirst(Position p) { return p.getNode() == header->next; }
    bool isLast(Position p) { return p.getNode() == trailer; }

    Position after(Position p) {
        if (isLast(p)) throw runtime_error("No position after last!");
        return Position(p.getNode()->next);
    }

    Position before(Position p) {
        if (isFirst(p)) throw runtime_error("No position before first!");
        return Position(p.getNode()->prev);
    }

    void insertFirst(string elem) { insertAt(first(), elem); }
    void insertLast(string elem) { insertAt(last(), elem); }

    string remove(Position p) {
        if (isLast(p)) throw runtime_error("Cannot remove trailer!");
        DListNode* node = p.getNode();
        string data = node->data;
        node->prev->next = node->next;
        node->next->prev = node->prev;
        delete node;
        count--;
        return data;
    }

    bool isEmpty() { return count == 0; }

    void display() {
        if (isEmpty()) {
            cout << "List is empty." << endl;
            return;
        }
        cout << "List contents: ";
        DListNode* curr = header->next;
        while (curr != trailer) {
            cout << curr->data;
            if (curr->next != trailer) cout << " <-> ";
            curr = curr->next;
        }
        cout << endl;
    }
};

//=================== MAIN ===================

int main() {
    VehicleRegistry registry;
    FourWayIntersection intersection;
    NavigationSystem<string> navigator;
    DoublyLinkedList accidents;

    int choice;

    while (true) {
        cout << "\n========== TRAFFIC MANAGEMENT SYSTEM ==========" << endl;
        cout << "1. Vehicle Registry (Hash Table)" << endl;
        cout << "2. Traffic Signals (Priority Queue)" << endl;
        cout << "3. Route Navigation (Two Stacks)" << endl;
        cout << "4. Accident Tracking (Doubly Linked List)" << endl;
        cout << "5. Exit" << endl;
        cout << "Enter choice: ";
        cin >> choice;

        if (choice == 1) {
            cout << "\n--- Vehicle Registry ---" << endl;
            cout << "1. Register  2. Search  3. Check Load Factor" << endl;
            int sub; cin >> sub;

            if (sub == 1) {
                string plate, owner, type, date;
                cout << "Enter plate: "; cin >> plate;
                cout << "Enter owner: "; cin >> owner;
                cout << "Enter type: "; cin >> type;
                cout << "Enter date: "; cin >> date;
                registry.registerVehicle(plate, VehicleInfo(owner, type, date));
                cout << "Vehicle registered." << endl;
            }
            else if (sub == 2) {
                string plate; cout << "Enter plate: "; cin >> plate;
                VehicleInfo* v = registry.searchVehicle(plate);
                if (v) cout << "Found: " << v->owner << ", " << v->type << endl;
                else cout << "Not found!" << endl;
            }
            else if (sub == 3) registry.displayLoadFactor();
        }
        else if (choice == 2) {
            cout << "\n--- Traffic Signals ---" << endl;
            cout << "1. Add Vehicle  2. Process All Directions  3. Display All" << endl;
            int sub; cin >> sub;

            if (sub == 1) {
                string dir, id; int type, arrival;
                cout << "Direction (NORTH/SOUTH/EAST/WEST): "; cin >> dir;
                cout << "Vehicle ID: "; cin >> id;
                cout << "Type (0=Normal, 1=Emergency): "; cin >> type;
                cout << "Arrival time: "; cin >> arrival;
                intersection.addVehicle(dir, id, (VehicleType)type, arrival);
            }
            else if (sub == 2) intersection.processAllDirections();
            else if (sub == 3) intersection.displayAll();
        }
        else if (choice == 3) {
            cout << "\n--- Route Navigation ---" << endl;
            cout << "1. Go Forward  2. Go Back  3. Show Route  4. Redo" << endl;
            int sub; cin >> sub;

            if (sub == 1) {
                string loc; cout << "Enter location: "; cin >> loc;
                navigator.goForward(loc);
            }
            else if (sub == 2) navigator.goBack();
            else if (sub == 3) navigator.showCurrentRoute();
            else if (sub == 4) navigator.undoLastMove();
        }
        else if (choice == 4) {
            cout << "\n--- Accident Tracking ---" << endl;
            cout << "1. Add Accident  2. Remove First  3. Display All" << endl;
            int sub; cin >> sub;

            try {
                if (sub == 1) {
                    string accident;
                    cout << "Enter accident details (lat_lon_time_severity): "; cin >> accident;
                    accidents.insertLast(accident);
                    cout << "Accident added." << endl;
                }
                else if (sub == 2) {
                    if (!accidents.isEmpty()) {
                        string removed = accidents.remove(accidents.first());
                        cout << "Removed: " << removed << endl;
                    } else cout << "List is empty!" << endl;
                }
                else if (sub == 3) accidents.display();
            } catch (runtime_error& e) { cout << "Error: " << e.what() << endl; }
        }
        else if (choice == 5) break;
    }

    return 0;
}
