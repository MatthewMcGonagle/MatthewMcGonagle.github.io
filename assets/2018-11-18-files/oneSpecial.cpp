/**
    Use a friend class template and template specialization to implement a class template that takes 
    a datatype and a qualifier; the qualifier comes in two states: standard or special. The class 
    template should have the same interface but two different behaviors for a member function 
    bar() based on the qualifier. 

    To minimize rewriting code, the behavior of bar() is implemented in a separate class template
    _Bar which is made a friend of Foo. Then _Bar::bar() is called from inside
    Foo::bar(). 

*/

#include <iostream>

enum Qualifier {standard, special};

// Need to declare the class template to use with its friend.
template <typename dataType, Qualifier qual> class _Bar;

/**
    The class template Foo has friend class template _Bar.
    The partial specializations are defined in _Bar. Also, the _Bar template is a friend of Foo.

    @tparam dataType The type of _myVar that the class holds. This will be printed
                     out using bar().
    @tparam qual The qualifier to specify whether to use standard behavior or special
                 behavior for Foo::Bar.
*/
template <typename dataType, Qualifier qual = standard>
class Foo {

    friend class _Bar<dataType, qual>;

    public:
    Foo(dataType myVar_) : _myVar(myVar_) {}

    /** 
        The behavior of bar() is defined by the version of bar() given inside the _Bar template.
    */   
    void bar() { _Bar<dataType, qual>::bar(this); }

    /**
        Calls bar() n times, outputting the count each time.
    */
    void bar(int n) {
        for(int i = 0; i < n; i++) {
            std::cout << "i = " << i << "\t"; 
            bar();
        }
    }
    
    private:
    dataType _myVar;
}; 

/**
    Structure template that creates the function bar(). It is a friend of Foo so that
    it can access the members of Foo.

    The general template is for the standard behavior given by passing qual = standard.
    @tparam dataType The datatype of Foo::_myVar that will be printed.
    @tparam qual Whether to use the standard behavior or the special behavior.
*/
template <typename dataType, Qualifier qual>
class _Bar{

    friend class Foo<dataType, qual>;

    private:
    /**
        Standard behavior for bar().        
        @param my Pointer to the instance of Foo<dataType, qual> that called bar(), e.g. the 
                  Foo object's this pointer.
    */
    static void bar(Foo<dataType, qual>* my) {
        std::cout << "Standard x = " << my->_myVar <<std::endl;
    }
};

/**
    This specialization for qual = special gives the special behavior of bar().

    @tparam dataType The type of Foo::_myVar that will be printed out.
*/
template <typename dataType>
class _Bar<dataType, special> {
    
    friend class Foo<dataType, special>;

    private:

    /**
        Special behavior for bar().
        @param my Pointer to the instance of Foo<dataType, qual> that called bar(), e.g. the
                  Foo object's this pointer.
    */
    static void bar(Foo<dataType, special>* my) {
        std::cout << "Special x = " << my->_myVar << std::endl;
    }
};

int main() {

    Foo<int> standardX(1);
    Foo<int, special> specialX(2);

    // Test out Foo::bar() for each object.

    std::cout << "standardX.bar()" << std::endl;
    standardX.bar();
    std::cout << "\nspecialX.bar()" << std::endl;
    specialX.bar();

    // Test out Foo::bar(int) for each object.

    std::cout << "\nstandardX.bar(2)" << std::endl;
    standardX.bar(2);
    std::cout << "\nspecialX.bar(3)" << std::endl;
    specialX.bar(3);

    return 0;
}
