int main() {
    volatile int a = 0;
    
    // BRANCH 1: Loop Condition
    // Executes 10,000 times. Taken 9,999 times (not taken the 10000th time).
    for (int i = 0; i < 10000; i++) {
        
        // BRANCH 2: Modulo Pattern
        // Executes 10,000 times.
        // Pattern: T, NT, NT, NT, T, NT, NT, NT...
        // The PAp predictor uses 6 bits of history. Because the pattern 
        // repeats every 4 branches, the 6-bit history should do well.
        if (i % 4 == 0) {
            a = 10;
        }
    }
    return 0;
}
