
/*
 * Question 4: Online Auction Platform
 * Part A: Hash Table with Multiplication Method (Fibonacci Hashing)
 * Part B: Vector with Binary Search (Active Auctions)
 * Part C: Stack with Validation (Bid History)
 * Part D: N-ary Tree with First-Child/Next-Sibling (Categories)
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
using namespace std;

//=================== PART A: HASH TABLE WITH MULTIPLICATION METHOD ===================

struct UserProfile {
    string passwordHash;
    string email;
    double rating;
    string joinDate;

    UserProfile() : rating(0.0) {}
    UserProfile(string ph, string e, string d) : passwordHash(ph), email(e), rating(5.0), joinDate(d) {}
};

struct UserNode {
    string username;
    UserProfile profile;
    UserNode* next;

    UserNode(string u, UserProfile p) : username(u), profile(p), next(nullptr) {}
};

class UserDatabase {
private:
    UserNode** table;
    int capacity;
    int size;
    const double A = 0.6180339887498949;  // Golden ratio conjugate

    // Horner's rule for string hash code
    long long hornerHash(string key) {
        long long hash = 0;
        for (char c : key) {
            hash = (hash * 31 + c);
        }
        return hash;
    }

    // Multiplication method: floor(m * (kA mod 1))
    // This is Fibonacci hashing (Knuth's recommendation)
    int multiplicationHash(string key) {
        long long k = hornerHash(key);
        // kA mod 1 = fractional part of k*A
        double product = k * A;
        double frac = product - floor(product);
        return (int)(capacity * frac);
    }

public:
    UserDatabase(int cap = 64) : capacity(cap), size(0) {
        table = new UserNode*[capacity];
        for (int i = 0; i < capacity; i++) table[i] = nullptr;
        cout << "[Using Fibonacci Hashing with A = (sqrt(5)-1)/2]" << endl;
    }

    void registerUser(string username, string password, string email, string date) {
        // Check if exists
        if (searchUser(username) != nullptr) {
            cout << "Username already exists!" << endl;
            return;
        }

        int idx = multiplicationHash(username);
        string passHash = to_string(hornerHash(password));  // Simple hash

        UserNode* newNode = new UserNode(username, UserProfile(passHash, email, date));

        // Insert at head (unsorted chaining for simplicity)
        newNode->next = table[idx];
        table[idx] = newNode;
        size++;

        cout << "User registered at index " << idx << ". Load factor: " << (double)size/capacity << endl;
    }

    bool authenticateUser(string username, string password) {
        int idx = multiplicationHash(username);
        UserNode* curr = table[idx];
        string passHash = to_string(hornerHash(password));

        while (curr != nullptr) {
            if (curr->username == username) {
                if (curr->profile.passwordHash == passHash) {
                    cout << "Authentication successful!" << endl;
                    return true;
                } else {
                    cout << "Invalid password!" << endl;
                    return false;
                }
            }
            curr = curr->next;
        }
        cout << "User not found!" << endl;
        return false;
    }

    void updateRating(string username, double newRating) {
        UserProfile* user = searchUser(username);
        if (user) {
            user->rating = newRating;
            cout << "Rating updated to " << newRating << endl;
        } else cout << "User not found!" << endl;
    }

    UserProfile* searchUser(string username) {
        int idx = multiplicationHash(username);
        UserNode* curr = table[idx];
        while (curr != nullptr) {
            if (curr->username == username) return &(curr->profile);
            curr = curr->next;
        }
        return nullptr;
    }

    void displayStats() {
        cout << "\n=== Hash Table Stats ===" << endl;
        cout << "Capacity: " << capacity << ", Size: " << size << endl;
        cout << "Load factor: " << (double)size/capacity << endl;
        cout << "Using golden ratio conjugate A = " << A << endl;

        // Show distribution
        int maxChain = 0;
        for (int i = 0; i < capacity; i++) {
            int chainLen = 0;
            UserNode* curr = table[i];
            while (curr != nullptr) { chainLen++; curr = curr->next; }
            maxChain = max(maxChain, chainLen);
        }
        cout << "Maximum chain length: " << maxChain << endl;
    }
};

//=================== PART B: VECTOR WITH BINARY SEARCH ===================

struct Auction {
    int itemID;
    string itemName;
    double currentBid;
    string highestBidder;
    int endTime;

    Auction(int id, string name, double bid, string bidder, int end) 
        : itemID(id), itemName(name), currentBid(bid), highestBidder(bidder), endTime(end) {}

    bool operator<(const Auction& other) const {
        return itemID < other.itemID;  // For sorting
    }
};

class AuctionVector {
private:
    Auction** arr;
    int capacity;
    int size;

    void resize() {
        int newCap = capacity * 2;
        Auction** newArr = new Auction*[newCap];
        for (int i = 0; i < size; i++) newArr[i] = arr[i];
        delete[] arr;
        arr = newArr;
        capacity = newCap;
        cout << "[Vector resized to " << capacity << "]" << endl;
    }

    // Binary search by itemID (Lec1/Lec4)
    int binarySearch(int itemID) {
        int left = 0, right = size - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid]->itemID == itemID) return mid;
            else if (arr[mid]->itemID < itemID) left = mid + 1;
            else right = mid - 1;
        }
        return -1;  // Not found
    }

    // Insertion sort for maintaining sorted order
    void insertionSort() {
        for (int i = 1; i < size; i++) {
            Auction* key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j]->itemID > key->itemID) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }

public:
    AuctionVector(int cap = 10) : capacity(cap), size(0) {
        arr = new Auction*[capacity];
    }

    void addAuction(int id, string name, double bid, string bidder, int end) {
        if (size == capacity) resize();

        // Check if exists
        if (binarySearch(id) != -1) {
            cout << "Auction with ID " << id << " already exists!" << endl;
            return;
        }

        arr[size++] = new Auction(id, name, bid, bidder, end);
        insertionSort();  // Maintain sorted order for binary search
        cout << "Auction " << id << " added." << endl;
    }

    Auction* findAuction(int itemID) {
        int idx = binarySearch(itemID);
        if (idx != -1) return arr[idx];
        return nullptr;
    }

    void updateBid(int itemID, double newBid, string bidder) {
        Auction* a = findAuction(itemID);
        if (a) {
            if (newBid > a->currentBid) {
                a->currentBid = newBid;
                a->highestBidder = bidder;
                cout << "Bid updated for item " << itemID << ": " << newBid << " by " << bidder << endl;
            } else {
                cout << "Bid must be higher than current " << a->currentBid << endl;
            }
        } else cout << "Auction not found!" << endl;
    }

    void removeAuction(int itemID) {
        int idx = binarySearch(itemID);
        if (idx == -1) {
            cout << "Auction not found!" << endl;
            return;
        }
        delete arr[idx];
        for (int i = idx; i < size - 1; i++) arr[i] = arr[i + 1];
        size--;
        cout << "Auction " << itemID << " removed." << endl;
    }

    void display() {
        cout << "\n=== Active Auctions (sorted by ID) ===" << endl;
        for (int i = 0; i < size; i++) {
            cout << "ID: " << arr[i]->itemID << ", Name: " << arr[i]->itemName 
                 << ", Current: $" << arr[i]->currentBid 
                 << ", By: " << arr[i]->highestBidder << endl;
        }
    }

    // Demonstrate amortized O(1) insertion, O(log n) search
    void demonstrateComplexity() {
        cout << "\n=== Complexity Analysis ===" << endl;
        cout << "Insertion: Amortized O(1) due to geometric resizing" << endl;
        cout << "Search: O(log n) using binary search on sorted array" << endl;
        cout << "Current size: " << size << ", Capacity: " << capacity << endl;
    }
};

//=================== PART C: STACK FOR BID HISTORY ===================

struct BidRecord {
    string bidderID;
    double amount;
    int timestamp;

    BidRecord(string id, double amt, int time) : bidderID(id), amount(amt), timestamp(time) {}
};

class BidHistoryStack {
private:
    BidRecord** arr;
    int capacity;
    int top;
    double reservePrice;

    void resize() {
        int newCap = capacity * 2;
        BidRecord** newArr = new BidRecord*[newCap];
        for (int i = 0; i <= top; i++) newArr[i] = arr[i];
        delete[] arr;
        arr = newArr;
        capacity = newCap;
    }

public:
    BidHistoryStack(double reserve, int cap = 10) : reservePrice(reserve), capacity(cap), top(-1) {
        arr = new BidRecord*[capacity];
    }

    ~BidHistoryStack() {
        for (int i = 0; i <= top; i++) delete arr[i];
        delete[] arr;
    }

    bool isEmpty() { return top == -1; }

    // Validate before pushing
    bool isBidValid(double amount) {
        if (amount < reservePrice) {
            cout << "Bid below reserve price of " << reservePrice << endl;
            return false;
        }
        if (!isEmpty() && amount <= arr[top]->amount) {
            cout << "Bid must exceed current highest of " << arr[top]->amount << endl;
            return false;
        }
        return true;
    }

    void placeBid(string bidderID, double amount, int timestamp) {
        if (!isBidValid(amount)) return;

        if (top == capacity - 1) resize();
        arr[++top] = new BidRecord(bidderID, amount, timestamp);
        cout << "Bid placed: $" << amount << " by " << bidderID << endl;
    }

    BidRecord* getLastBid() {
        if (isEmpty()) throw runtime_error("No bids yet!");
        return arr[top];
    }

    void undoBid() {
        if (isEmpty()) {
            cout << "Nothing to undo!" << endl;
            return;
        }
        BidRecord* bid = arr[top--];
        cout << "Undid bid of $" << bid->amount << " by " << bid->bidderID << endl;
        delete bid;
    }

    // Determine winner by popping all and finding highest
    string determineWinner() {
        if (isEmpty()) return "No bids placed";

        double maxBid = 0;
        string winner;

        // Pop all to find winner
        while (!isEmpty()) {
            BidRecord* bid = arr[top--];
            if (bid->amount > maxBid) {
                maxBid = bid->amount;
                winner = bid->bidderID;
            }
            delete bid;
        }

        cout << "\nAuction ended! Winner: " << winner << " with $" << maxBid << endl;
        return winner;
    }

    void displayHistory() {
        if (isEmpty()) {
            cout << "No bid history." << endl;
            return;
        }
        cout << "Bid history (newest first):" << endl;
        for (int i = top; i >= 0; i--) {
            cout << "  $" << arr[i]->amount << " by " << arr[i]->bidderID 
                 << " at t=" << arr[i]->timestamp << endl;
        }
    }
};

//=================== PART D: N-ARY TREE (FIRST-CHILD/NEXT-SIBLING) ===================

struct CategoryNode {
    string name;
    CategoryNode* firstChild;    // Leftmost child
    CategoryNode* nextSibling;   // Right sibling
    vector<string> items;

    CategoryNode(string n) : name(n), firstChild(nullptr), nextSibling(nullptr) {}
};

class CategoryTree {
private:
    CategoryNode* root;

    CategoryNode* findCategory(CategoryNode* node, string name) {
        if (node == nullptr) return nullptr;
        if (node->name == name) return node;

        // Search children
        CategoryNode* found = findCategory(node->firstChild, name);
        if (found) return found;

        // Search siblings
        return findCategory(node->nextSibling, name);
    }

    // Preorder traversal: Root -> First Child -> Next Sibling
    void preorderRec(CategoryNode* node, int depth) {
        if (node == nullptr) return;

        string indent(depth * 2, ' ');
        cout << indent << "- " << node->name;
        if (!node->items.empty()) {
            cout << " [" << node->items.size() << " items]";
        }
        cout << endl;

        preorderRec(node->firstChild, depth + 1);   // Visit children
        preorderRec(node->nextSibling, depth);      // Visit siblings
    }

    int countCategories(CategoryNode* node) {
        if (node == nullptr) return 0;
        return 1 + countCategories(node->firstChild) + countCategories(node->nextSibling);
    }

    int treeHeight(CategoryNode* node) {
        if (node == nullptr) return -1;
        int childHeight = treeHeight(node->firstChild);
        return 1 + childHeight;  // Height is max depth from root
    }

public:
    CategoryTree() {
        root = new CategoryNode("All Categories");
    }

    void addCategory(string parentName, string newCategory) {
        CategoryNode* parent = findCategory(root, parentName);
        if (!parent) {
            cout << "Parent category not found!" << endl;
            return;
        }

        CategoryNode* newNode = new CategoryNode(newCategory);

        // Add as first child or next sibling
        if (parent->firstChild == nullptr) {
            parent->firstChild = newNode;
        } else {
            // Add as last sibling
            CategoryNode* sibling = parent->firstChild;
            while (sibling->nextSibling != nullptr) sibling = sibling->nextSibling;
            sibling->nextSibling = newNode;
        }
        cout << "Added " << newCategory << " under " << parentName << endl;
    }

    void addItemToCategory(string category, string item) {
        CategoryNode* node = findCategory(root, category);
        if (node) {
            node->items.push_back(item);
            cout << "Added item to " << category << endl;
        } else cout << "Category not found!" << endl;
    }

    void displayHierarchy() {
        cout << "\n=== Category Hierarchy (Preorder) ===" << endl;
        preorderRec(root, 0);
    }

    void displayStats() {
        cout << "Total categories: " << countCategories(root) << endl;
        cout << "Tree height: " << treeHeight(root) << endl;
    }

    void listItems(string category) {
        CategoryNode* node = findCategory(root, category);
        if (node && !node->items.empty()) {
            cout << "Items in " << category << ": ";
            for (string& item : node->items) cout << item << " ";
            cout << endl;
        } else cout << "No items found in " << category << endl;
    }
};

//=================== MAIN ===================

int main() {
    UserDatabase users;
    AuctionVector auctions;
    BidHistoryStack bids(100.0);  // Reserve price $100
    CategoryTree categories;

    // Initialize sample categories
    categories.addCategory("All Categories", "Electronics");
    categories.addCategory("All Categories", "Fashion");
    categories.addCategory("Electronics", "Phones");
    categories.addCategory("Electronics", "Laptops");
    categories.addCategory("Phones", "Smartphones");

    int choice;

    while (true) {
        cout << "\n========== ONLINE AUCTION PLATFORM ==========" << endl;
        cout << "1. User Management (Hash Table)" << endl;
        cout << "2. Auction Management (Vector + Binary Search)" << endl;
        cout << "3. Bidding System (Stack)" << endl;
        cout << "4. Categories (N-ary Tree)" << endl;
        cout << "5. Exit" << endl;
        cout << "Enter choice: ";
        cin >> choice;

        if (choice == 1) {
            cout << "\n--- User Management ---" << endl;
            cout << "1. Register  2. Login  3. Update Rating  4. Stats" << endl;
            int sub; cin >> sub;

            string u, p, e, d;
            if (sub == 1) {
                cout << "Username: "; cin >> u;
                cout << "Password: "; cin >> p;
                cout << "Email: "; cin >> e;
                cout << "Date: "; cin >> d;
                users.registerUser(u, p, e, d);
            }
            else if (sub == 2) {
                cout << "Username: "; cin >> u;
                cout << "Password: "; cin >> p;
                users.authenticateUser(u, p);
            }
            else if (sub == 3) {
                double r; cout << "Username and new rating: "; cin >> u >> r;
                users.updateRating(u, r);
            }
            else if (sub == 4) users.displayStats();
        }
        else if (choice == 2) {
            cout << "\n--- Auction Management ---" << endl;
            cout << "1. Add Auction  2. Find  3. Update Bid  4. Remove  5. Display  6. Complexity" << endl;
            int sub; cin >> sub;

            if (sub == 1) {
                int id, end; string name, bidder; double bid;
                cout << "ID Name Bid Bidder EndTime: "; cin >> id >> name >> bid >> bidder >> end;
                auctions.addAuction(id, name, bid, bidder, end);
            }
            else if (sub == 2) {
                int id; cout << "Enter ID: "; cin >> id;
                Auction* a = auctions.findAuction(id);
                if (a) cout << "Found: " << a->itemName << ", $" << a->currentBid << endl;
                else cout << "Not found!" << endl;
            }
            else if (sub == 3) {
                int id; double bid; string bidder;
                cout << "ID NewBid Bidder: "; cin >> id >> bid >> bidder;
                auctions.updateBid(id, bid, bidder);
            }
            else if (sub == 4) { int id; cin >> id; auctions.removeAuction(id); }
            else if (sub == 5) auctions.display();
            else if (sub == 6) auctions.demonstrateComplexity();
        }
        else if (choice == 3) {
            cout << "\n--- Bidding System ---" << endl;
            cout << "1. Place Bid  2. Undo  3. History  4. End Auction (Determine Winner)" << endl;
            int sub; cin >> sub;

            try {
                if (sub == 1) {
                    string bidder; double amt; int time;
                    cout << "Bidder Amount Time: "; cin >> bidder >> amt >> time;
                    bids.placeBid(bidder, amt, time);
                }
                else if (sub == 2) bids.undoBid();
                else if (sub == 3) bids.displayHistory();
                else if (sub == 4) bids.determineWinner();
            } catch (runtime_error& e) { cout << "Error: " << e.what() << endl; }
        }
        else if (choice == 4) {
            cout << "\n--- Categories ---" << endl;
            cout << "1. Add Category  2. Add Item  3. Display Hierarchy  4. Stats  5. List Items" << endl;
            int sub; cin >> sub;

            string p, c, item;
            if (sub == 1) { cout << "Parent NewCategory: "; cin >> p >> c; categories.addCategory(p, c); }
            else if (sub == 2) { cout << "Category Item: "; cin >> c >> item; categories.addItemToCategory(c, item); }
            else if (sub == 3) categories.displayHierarchy();
            else if (sub == 4) categories.displayStats();
            else if (sub == 5) { cout << "Category: "; cin >> c; categories.listItems(c); }
        }
        else if (choice == 5) break;
    }

    return 0;
}
