/**
    templateTemplateNoCompile.cpp

    This is an example of the classes in templateTemplate.cpp not compiling if the super-class doesn't
    implement bar().
*/

#include <iostream>

/**
    Class template for standard bar().

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
    This is missing the member function bar().

    @tparam dataType The type of _myVar.
*/
template <typename dataType>
class NoBar {

    protected:

    dataType _myVar;

};

/**
    The class template Foo defaults to have the standard qualifier. The
    member function bar(int n) will make calls to the particular version of
    bar() created by the super-class template _Bar.

    @tparam dataType The datatype of Bar::_myVar.
    @tparam Bar The super class template implementing the member function bar() and
                holds the member variable _myVar.
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

int main() {

    Foo<int> standardX(1);
    Foo<int, NoBar> noCompileX(2);

}
