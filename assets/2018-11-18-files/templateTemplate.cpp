/**
    templateTemplate.cpp

    Use inheritance and a template template parameter to specify the behavior of Foo::bar().
 
    Now we pass in a class template that generates the behavior of bar(). The catch is that if the
    user passes in a class template that doesn't generate the member function bar() or hold 
    the member variable _myVar, then there will be compiler errors.
*/

#include <iostream>

/**
    Class template that implements the standard behavior of bar().

    @tparam dataType The type of _myVar.
*/

template <typename dataType>
class StandardBar {

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
    The class template giving the special version of bar().

    @tparam dataType The type of _myVar.
*/

template <typename dataType>
class SpecialBar {

    public:
    
    void bar() {

        std::cout << "Special _myVar = " << _myVar << std::endl;

    }

    protected:

    dataType _myVar;

};

/**
    The class template Foo defaults to have the standard version of bar(). The
    member function bar(int n) will make calls to the particular version of
    bar() created by the super-class template Bar.

    @tparam dataType The datatype of _Bar::_myVar.
    @tparam Bar The class template that gives the version of bar() and holds _myVar. Default
                is StandardBar. 
*/

template <typename dataType, template<typename> class Bar = StandardBar> 
class Foo : public Bar<dataType> {

    public:
    Foo(dataType myVar_) {

        // Note the need for Bar<dataType>:: to find the variable.
        Bar<dataType>::_myVar = myVar_;
    } 

    // Need to tell compiler to look for the other overloaded version of bar(). 
    using Bar<dataType>::bar; 

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

/**
    Create an alias to make using Foo<dataType, SpecialBar> easier on the user.
*/

template <typename dataType>
using FooSpecial = Foo<dataType, SpecialBar>;

int main() {

    Foo<int> standardX(1);
    FooSpecial<int> specialX(2);

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
