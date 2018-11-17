/**

    Use inheritance and template specialization to implement a class template that takes a datatype and a qualifier; 
    the qualifier comes in two states: standard or special. The class template should have the same interface 
    but two different behaviors for a member function bar() based on the qualifier. 

    The catch is that member function template partial specialization requires the class template partial
    specialization to also be defined. The use of inheritance allows us to minimize the need to rewrite code
    for the class template partial specialization. 
*/

#include <iostream>

enum Qualifier {standard, special};

/** 
    Class template _Bar holds the implementations of bar(). The general case
    is meant to be the standard case of the template.

    We put _myVar in _Bar since it is needed for function bar().

    @tparam dataType The type of _myVar, the variable held by _Bar.
    @tparam qualifier The qualifier as to whether to use the standard behavior or the
                 special behavior.
*/

template <typename dataType, Qualifier qualifier> 
class _Bar {

    public:

    /**
        The standard version of bar(). 
    */
    void bar() {

        std::cout << "Standard _myVar = " << _myVar << std::endl;
    }

    protected:

    dataType _myVar;

};

/**
    The partial specialization of class template _Bar that gives the
    special behavior of bar().

    @tparam dataType The datatype of _Bar::_myVar.
*/

template <typename dataType>
class _Bar<dataType, special> {

    public:
    
    void bar() {

        std::cout << "Special _myVar = " << _myVar << std::endl;

    }

    protected:

    dataType _myVar;
};

/**
    The class template Foo defaults to have the standard qualifier. The
    member function nBar() will make calls to the particular version of
    bar() created by the super-class template _Bar.

    @tparam dataType The datatype of _Bar::_myVar.
    @tparam qualifier The qualifier for whether to use standard behavior or
                 to use special behavior. Default: standard.
*/

template <typename dataType, Qualifier qual = standard>
class Foo : public _Bar<dataType, qual> {

    public:
    Foo(dataType myVar_) {
        // Note the need for _Bar<dataType, qual>:: to find the variable.
        _Bar<dataType, qual>::_myVar = myVar_;
    } 

    // Need to tell compiler to look for the other overloaded version of bar(). 
    using _Bar<dataType, qual>::bar; 

    /**
        Calls bar n times, each time printing out the count.

        @param n The number of times to call Foo::bar().
    */ 
    void bar(int n) {
        for(int i = 0; i < n; i++) {
            std::cout << "i = " << i << "\t";
            bar();
        }
    }
    
}; 

int main() {

    Foo<int> standardX(1);
    Foo<int, special> specialX(2);

    // Test out Foo::bar().

    std::cout << "standardX.bar()" << std::endl;
    standardX.bar();
    std::cout << "\nspecialX.bar()" << std::endl;
    specialX.bar();

    // Test out Foo::bar(int).

    std::cout << "\nstandardX.bar(2)" << std::endl;
    standardX.bar(2);
    std::cout << "\nspecialX.bar(3)" << std::endl;
    specialX.bar(3);

    return 0;
}
