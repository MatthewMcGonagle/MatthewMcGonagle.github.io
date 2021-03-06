---
layout: post
date: 2018-11-18
title: Templates with Special and Standard Behaviors 
tags: C++ Templates MetaProgramming
---

{% capture inheritanceContent %}
    {% include 2018-11-18-files/inheritance.cpp %}
{% endcapture %}

## [Download inheritance.cpp Here]({{site . url}}/assets/2018-11-18-files/inheritance.cpp)
## [Download oneSpecial.cpp Here]({{site . url}}/assets/2018-11-18-files/oneSpecial.cpp)
## [Download aliasingExample.cpp Here]({{site . url}}/assets/2018-11-18-files/aliasingExample.cpp)
## [Download templateTemplate.cpp Here]({{site . url}}/assets/2018-11-18-files/templateTemplate.cpp)
## [Download noCompile.cpp Here]({{site . url}}/assets/2018-11-18-files/noCompile.cpp)
## [Download templateTemplateNoCompile.cpp Here]({{site . url}}/assets/2018-11-18-files/templateTemplateNoCompile.cpp)


In this post we'll look at several ways to use template specialization (including partial specialization)
to give templates different behaviors depending on their inputs. I was inspired to look consider this while
working on the [penguinV](https://github.com/ihhub/penguinV); in that project there are classes whose
interface are the same but they have different behavior for whether they deal with memory on the hosting computer
memory or on a GPU. There are a small number of functions that deal with memory directly and many more functions
that need deal with memory indirectly by making calls to the former functions. Code repetition is reduced by
using dynamic resolution (i.e. at run time) of the functions that deal with memory directly. 

This post is the result of my wondering if there is a way to make this resolution statically, i.e. at compile
time. It turns out there are at least several ways to use C++ template metaprogramming to handle this
problem in such a manner. I will review the ones I consider elegant enough. 

Instead of looking at the issue of memory, we will look at creating a class template `Foo<typename dataType, 
Qualifier qualifier>`
that stores a variable of type `dataType`. The template parameter `qualifier` will be of enumerated type `Qualifier`
and can be either `standard` or `special`; furthermore, we will `qualifier` default to `standard`. The qualifier
will determine the behavior of the member function `Foo<dataType, qualifier>::bar()`. 

Finally there will be one implementation of `Foo<dataType, qualifier>::bar(int n)` that calls `Foo<dataType, qualifier>::bar()` n times.

The catch to keep in mind is that C++ does not allow you to create a partial template specialization of a member
function (such as `Foo<dataType, qualifier>::bar()`) without creating a separate definition for the partially
specialized class. We intend for every value of `qualifier` to give a `Foo` with the same member functions, 
but their behavior should be different. Keeping multiple copies of effectively the same class declaration
is certainly undesirable. 

In my opinion, the best way to handle the "multiple copies" issue is to use inheritance to reduce the size of each
copy. The copies are still there, but we can keep their size small. 

As an example of how this looks for a user of your classes, consider the main execution code from `inheritance.cpp`:
{% assign codeLines = inheritanceContent | newline_to_br | split: '<br />' %}
{% highlight cpp %}
// From inheritance.cpp 
{% for line in codeLines offset: 105 limit:21 %}{{ line }}{% endfor %}
{% endhighlight %}

This gives the output
{% highlight text %}
{% include 2018-11-18-files/out.inheritance %}
{% endhighlight %}

So we make it easy for users of our classes to decide at compile time whether to use the standard or special
behavior.

First, we'll look at why a naive simple solution may not work (won't even compile). 
Then we'll look at the inheritance solution, followed by some other solutions that in my opinion aren't as 
elegant.

# Different Typedefs Do NOT Work

My original inspiration from the penguinV project is for the case of two special behaviors: work with memory
on the hosting computer or work with memory on a GPU. For such a case, we really only want two different
behaviors, and you may be tempted to just define different behaviors for different uses of `typedef` (so for this
section we consider not using a type parameter `typename dataType` at all). 

However, remember that `typedef` really just creates type synonyms; so it may not be possible to define
different behavior for different instances of `typedef`. For example, consider the following code 
from `noCompile.cpp`; the compiler will have an error, because the the definitions of `Foo<standardInt>` and
`Foo<specialInt>` are really two different definitions of `Foo<int>`. 

{% highlight cpp %}
{% include 2018-11-18-files/noCompile.cpp %}
{% endhighlight %}

# Using Inheritance

Now we look at using partial template specialization in combination with inheritance to minimize the amount
of code we need to copy. We put the separate implementations of `bar()` inside a base class template
`_Bar<typename dataType, Qualifier qualifier>`. Here we are using the convention that the 
underscore in `_Bar` should tell the user that this class should be considered 
"private" to `inheritance.cpp`.

 Take note of the fact that we need to make separate 
definitions for the base class template `_Bar`; that is we need to specify more than just the 
partial specializations of the member function `_Bar<dataType, qualifier>::bar()`. However, the code copying
is minimal compared to copying all of `Foo`. 

{% highlight cpp %}
{% include 2018-11-18-files/inheritance.cpp %}
{% endhighlight %}

# Using Friend Classes

Another option is to use a friend class to implement the functions with different behavior. Now we use
a class template `_Bar` to hold the different versions of `bar()`. The catch is that we need to pass
an explicit pointer or reference to the class calling `_Bar::bar()`; so really, the parameter format is 
`Bar<dataType, qualifier>::bar(Foo<dataType, qualifier>*)`. 

Then, inside the class template `Foo` we can make a general member function `Foo::bar()` that will call 
the appropriate `_Bar<dataType, qualifier>::bar(Foo<dataType, qualifier> *)`. Of course, we will need
to make `_Bar` a friend of `Foo`. If we wish to restrict any other classes from calling `_Bar` then we
can make everything inside `_Bar` private and also make `Foo` a friend of `_Bar`. 

{% highlight cpp%}
{% include 2018-11-18-files/oneSpecial.cpp %}
{% endhighlight %}

# Using Aliasing For a Finite Number of Possibilities

What if we aren't interested in creating behavior for all possible outputs? For example how can achieve
standard behavior for `int` and `double`; special behavior for `int`; and no other behavior? A quick trick
is to use an enumerative type to determine the behavior; this allows us to get two different behaviors for
`int`. However, then we are left with the problem of generating the type of `_myVar` using a non-type 4
template parameter. This can be accomplished using aliasing. 

{% highlight cpp %}
{% include 2018-11-18-files/aliasingExample.cpp %}
{% endhighlight %}

# Using Template Template Parameters

The final possibility we list is using a template parameter that is itself a template (a so called template
template parameter). This example uses inheritance, but now we pass a class template to generate the
super class. However, the super class must implement the member function `bar()` and hold a variable
`_myVar`. If we pass a class template that fails to do so, then we will generate compiler errors. So in a
sense this method is less safe than the methods shown above since it is possible for the user to 
generate code that won't compile. However, this can be avoided by proper documentation. 

{% highlight cpp %}
{% include 2018-11-18-files/templateTemplate.cpp %}
{% endhighlight %}

Let's take a look at an example where compilation will fail if we don't implement the super class template
correctly. The problem with the following code is that the class template `NoBar` doesn't implement the
member function `bar()`.

{% highlight cpp %}
{% include 2018-11-18-files/templateTemplateNoCompile.cpp %}
{% endhighlight %}

When we try to compile this with `g++ templateTemplateNoCompile.cpp -std=c++11`, we get the following
error messages:
```
templateTemplateNoCompile.cpp: In instantiation of 'class Foo<int, NoBar>':
templateTemplateNoCompile.cpp:87:31:   required from here
templateTemplateNoCompile.cpp:68:26: error: no members matching 'NoBar<int>::bar' in 'class NoBar<int>'
     using Bar<dataType>::bar;
                          ^
``` 

## [Download inheritance.cpp Here]({{site . url}}/assets/2018-11-18-files/inheritance.cpp)
## [Download oneSpecial.cpp Here]({{site . url}}/assets/2018-11-18-files/oneSpecial.cpp)
## [Download aliasingExample.cpp Here]({{site . url}}/assets/2018-11-18-files/aliasingExample.cpp)
## [Download templateTemplate.cpp Here]({{site . url}}/assets/2018-11-18-files/templateTemplate.cpp)
## [Download noCompile.cpp Here]({{site . url}}/assets/2018-11-18-files/noCompile.cpp)
## [Download templateTemplateNoCompile.cpp Here]({{site . url}}/assets/2018-11-18-files/templateTemplateNoCompile.cpp)

