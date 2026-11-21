using LMSharp.Gguf;
using LMSharp.Tokenization;
using Xunit;

namespace LMSharp.UnitTests;

public sealed class Phi4TokenizerTests
{
    [Fact]
    public void FormatsTextOnlyPrompts()
    {
        Assert.Equal("<|user|>Hello<|end|><|assistant|>", Phi4Tokenizer.FormatPrompt("Hello"));
        Assert.Equal("<|system|>Be brief.<|end|><|user|>Hello<|end|><|assistant|>",
            Phi4Tokenizer.FormatPrompt("Hello", "Be brief."));
    }

    [Phi4ModelFact]
    public async Task MatchesPublishedTokenizer()
    {
        var model = await new GgufReader().ReadAsync(Phi4ModelFactAttribute.ModelPath);
        var tokenizer = new Phi4Tokenizer(model);
        Assert.Equal(200064, tokenizer.VocabularySize);
        Assert.Equal(199999, tokenizer.EndOfTextId);
        Assert.Equal(200020, tokenizer.EndOfTurnId);

        // Hugging Face tokenizers 0.23.3 with Microsoft's local tokenizer.json.
        (string Text, int[] Ids)[] roundTrips =
        [
            ("", []),
            ("Hello, world!", [13225, 11, 2375, 0]),
            ("  hello\tworld \r\n  ", [220, 40617, 156300, 4568, 256]),
            ("We're DON'T HTTPServer foo12345", [48503, 153384, 21929, 6444, 30551, 7633, 2548]),
            ("Doſn't a'ſ", [6449, 70067, 3023, 261, 6, 70067]),
            ("café e\u0301 中文 العربية", [66, 103112, 319, 13430, 83711, 33968]),
            ("one\u00A0two\u2003\n", [690, 5310, 38397, 33203, 198]),
            ("👍🏽 👩\u200D💻", [82514, 52622, 121, 61138, 102, 2524, 31446, 119]),
            ("😀🧪 𐐀𐐨𐐷 𠮷 𝟙𝟚𝟛𝟜",
                [84083, 4103, 100, 103, 220, 172, 238, 238, 222, 172, 238, 238, 101, 172, 238, 238, 115,
                 220, 172, 254, 106, 115, 220, 43120, 253, 247, 43120, 253, 248, 43120, 253, 249, 43120, 253, 250]),
        ];

        foreach (var (text, ids) in roundTrips)
        {
            Assert.Equal(ids, tokenizer.Encode(text));
            Assert.Equal(text, tokenizer.Decode(ids));
        }

        Assert.Equal([200019, 3686, 200020], tokenizer.Encode("<|assistant|>  \n hi<|end|>"));
        Assert.Equal("hi", tokenizer.Decode([200019, 3686, 200020]));
        Assert.Equal("<|assistant|>hi<|end|>", tokenizer.Decode([200019, 3686, 200020], skipSpecialTokens: false));
        Assert.Equal([200023, 39859, 200024], tokenizer.Encode("<|tool|> \u2003payload<|/tool|>\t"));
        Assert.Equal("<|tool|>payload<|/tool|>", tokenizer.Decode([200023, 39859, 200024]));
        Assert.Equal([87, 58, 92158, 1179, 42363, 60, 220, 199999, 342],
            tokenizer.Encode("x[PAD200029] <|endoftext|> y"));
        Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.Decode([200064]));

        // Jinja2 3.1.6 renders tokenizer_config.json with add_generation_prompt = true.
        Assert.Equal([200021, 4827, 382, 220, 17, 659, 220, 17, 30, 200020, 200019],
            tokenizer.Encode(Phi4Tokenizer.FormatPrompt("What is 2 + 2?")));
        Assert.Equal([200022, 3238, 82463, 13, 200020, 200021, 62316, 5911, 13, 200020, 200019],
            tokenizer.Encode(Phi4Tokenizer.FormatPrompt("  Say hi.", "Be concise.")));
    }
}
