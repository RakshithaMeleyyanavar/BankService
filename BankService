public class BankService { 
 
    // Nested class (can be public or private, doesn't matter for functionality) 
    static class BankAccount { 
        private double balance; 
 
        public BankAccount(double initialBalance) { 
            if (initialBalance < 0) { 
                throw new IllegalArgumentException("Initial balance cannot be negative"); 
            } 
            this.balance = initialBalance; 
        } 
 
        public void deposit(double amount) { 
            if (amount <= 0) { 
                throw new IllegalArgumentException("Deposit must be positive"); 
            } 
            balance += amount; 
        } 
 
        public void withdraw(double amount) { 
            if (amount <= 0 || amount > balance) { 
                throw new IllegalArgumentException("Invalid withdrawal"); 
            } 
            balance -= amount; 
        } 
 
        public double getBalance() { 
            return balance; 
        } 
    } 
 
    // The main entry point 
    public static void main(String[] args) { 
        // Create an instance of the BankAccount class 
        BankAccount acc = new BankAccount(5000); 
         
        System.out.println("Initial Balance: " + acc.getBalance()); 
 
        // Perform operations 
        acc.deposit(700); 
        System.out.println("Balance after deposit of 500: " + acc.getBalance()); 
 
        acc.withdraw(100); 
        System.out.println("Balance after withdrawal of 300: " + acc.getBalance()); 
 
        // Print final result 
        System.out.println("Final Balance: " + acc.getBalance()); 
    } 
} 
