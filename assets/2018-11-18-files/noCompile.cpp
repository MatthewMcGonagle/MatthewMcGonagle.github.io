/**
    noCompile.cpp

    Example that shows how just using typedef's for different behavior of using
    int's will NOT compile. Typedef's just provide type synonyms; so by defining
    Foo<standardInt> and Foo<specialInt>, our example is trying to give two different
    definitions for Foo<int>::bar().

*/

#include <iostream>

typedef int standardInt;
typedef int specialInt;

/**
    Class template for different behavior for dealing with integers. 
    @param dataType the dataType the class will be dealing with.
*/
template <typename dataType>
class Foo {

    public:
    Foo(dataType myVar_) : _myVar(myVar_) {}
  
    /**
        Behavior of bar() should depend on dataType.
    */ 
    void bar();
    
    private:
    dataType _myVar; 
}; 

/**
    For type standardInt, bar() should just print adding 1 to _myVar.   
*/
template <>
void Foo<standardInt>::bar() {
    std::cout << "standardInt _myVar = " << _myVar << std::endl;
}

/**
    For type specialInt, bar() should just print multiplying _myVar by 3. 

    BUT THERE IS A PROBLEM! The names specialInt and standardInt are just other names for
    the type int. So we are trying to the the template two different definitions for 
    template parameter int, i.e. two different defintions of Foo<int>::bar().

    So this will NOT compile!
*/
template <>
void Foo<specialInt>::bar() {
    std::cout << "specialInt _myVar = " _myVar << std::endl; 
}

typedef Foo<standardInt> FooInt; /**< A special typedef to hide the template usage. */

int main() {

    Foo<standardInt> x(1);
    FooInt x2(1);
    Foo<specialInt> y(1);

    std::cout << "x.bar()" << std::endl;
    x.bar();

    std::cout << "\nx2.bar()" << std::endl;
    x2.bar();

    std::cout << "\nx3.bar()" << std::endl;
    y.bar();

    return 0;
}
