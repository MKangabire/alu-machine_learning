function outer() {
    let count = 0;
    return function inner() {
        count++;
        return count;
    };
}
const increment = outer();
console.log(increment()); // 1
console.log(increment()); // 2
