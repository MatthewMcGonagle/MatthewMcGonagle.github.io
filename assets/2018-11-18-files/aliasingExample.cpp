/**
   Restrict possible outputs of Foo class tempalte to a finite number by using a non-type parameter, in particular 
   use values of a special enumerative type. 
*/

#include <iostream>

enum DataTypeId {standardInt, specialInt, standardDouble};

/** A structure template to hold the type of data to hold inside the template class Foo.
    Will be used with aliasing.
*/
template <DataTypeId Id> struct DataTypeHolder;

template <>
struct DataTypeHolder<standardInt> {
    typedef int type; 
};

template <>
struct DataTypeHolder<specialInt> {
    typedef int type;
};

template <>
struct DataTypeHolder<standardDouble> {
    typedef double type;   
};

/**
    Alisaing tempalte that gives the data type for Foo::_myVar.
*/
template <DataTypeId Id>
using _myVarType = typename DataTypeHolder<Id>::type;

template <DataTypeId Id>
class Foo {

    // For shorthand, can use typedef to make shortcut for _myVarType<Id>.
    typedef _myVarType<Id> datatype;

    public:
    Foo(datatype myVar_) : _myVar(myVar_) {}

    /**
        The behavior of bar() will depend on the template parameter Id.
        No matter what Id is, bar() will print out the value of _myVar, but
        it will also print out a message depending on the value of Id.
    */   
    void bar();

    /**
        Calls bar() n times, each time printing out the count.

        @param n The number of times to call bar().
    */
    void bar(int n);
    
    private:
    datatype _myVar;
}; 

template <DataTypeId Id>
void Foo<Id>::bar(int n) {

    for (int i = 0; i < n; i++) {
        std::cout << "i = " << i << "\t";    
        bar();
    }
}


template <>
void Foo<standardInt>::bar() {
 std::cout << "standardInt " << _myVar << std::endl;
}

template <>
void Foo<specialInt>::bar() {
    std::cout << "specialInt " << _myVar << std::endl;
}

template <>
void Foo<standardDouble>::bar() {
    std::cout << "standardDouble " << _myVar << std::endl;
}

// One can use typedef to make shorthand versions of each type if needed/desired.
typedef Foo<standardInt> FooInt;

int main() {

    Foo<standardInt> x(1);
    FooInt x2(2);
    Foo<specialInt> y(3);
    Foo<standardDouble> z(4.5);

    std::cout << "x.bar()" <<std::endl;
    x.bar();
    std::cout << "\nx2.bar()" << std::endl;
    x2.bar();
    std::cout << "\ny.bar()" << std::endl;
    y.bar();
    std::cout << "\nz.bar()" << std::endl;
    z.bar();

    std::cout << "\nx.bar(5)" << std::endl;
    x.bar(5);

    return 0;
}
