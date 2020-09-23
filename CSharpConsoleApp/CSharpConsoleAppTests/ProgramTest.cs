using CSharpConsoleApp;
using NUnit.Framework;

namespace CSharpConsoleAppTests
{
    public class ProgramTest
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Create()
        {
            Program program = new Program();
            Assert.Pass();
        }
    }
}