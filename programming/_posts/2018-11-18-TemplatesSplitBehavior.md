---
layout: post
date: 2018-11-16
title: Templates with Special and Standard Behaviors 
tags: C++ Templates MetaProgramming
---
{% capture inheritanceContent %}
    {% include 2018-11-18-files/inheritance.cpp %}
{% endcapture %}

## [Download inheritance.cpp Here]({{site . url}}/assets/2018-11-18-files/inheritance.cpp)

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
