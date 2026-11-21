using System;
using Xunit;

namespace LMSharp.UnitTests
{
    public class ProgramTests
    {
        [Fact]
        public void VersionFlagReturnsVersion()
        {
            var output = LMSharp.Console.Program.Run(["--version"]);
            Assert.Equal("1.0.0", output);
        }

        [Fact]
        public void NoFlagReturnsUsage()
        {
            var output = LMSharp.Console.Program.Run(Array.Empty<string>());
            Assert.StartsWith("Usage", output);
        }
    }
}
