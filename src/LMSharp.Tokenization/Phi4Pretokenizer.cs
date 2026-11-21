using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace LMSharp.Tokenization;

internal static partial class Phi4Pretokenizer
{
    // The gpt-4o pattern from the published Phi-4 Mini tokenizer.json.
    [GeneratedRegex("""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+""", RegexOptions.CultureInvariant, 5000)]
    private static partial Regex Pattern();

    public static IEnumerable<string> Split(string text)
    {
        var regexInput = text;
        List<int>? utf16Offsets = null;
        if (text.Any(char.IsSurrogate))
        {
            // .NET regex classifies UTF-16 code units. Match one BMP representative per
            // supplementary rune, then use the original text for byte encoding.
            var projected = new StringBuilder(text.Length);
            utf16Offsets = new List<int>(text.Length + 1);
            var offset = 0;
            foreach (var rune in text.EnumerateRunes())
            {
                utf16Offsets.Add(offset);
                projected.Append(rune.IsBmp ? (char)rune.Value : CategoryRepresentative(rune));
                offset += rune.Utf16SequenceLength;
            }

            utf16Offsets.Add(offset);
            regexInput = projected.ToString();
        }

        foreach (Match match in Pattern().Matches(regexInput))
        {
            var start = utf16Offsets?[match.Index] ?? match.Index;
            var end = utf16Offsets?[match.Index + match.Length] ?? match.Index + match.Length;
            yield return text[start..end];
        }
    }

    private static char CategoryRepresentative(Rune rune) => Rune.GetUnicodeCategory(rune) switch
    {
        UnicodeCategory.UppercaseLetter => 'A',
        UnicodeCategory.LowercaseLetter => 'b',
        UnicodeCategory.TitlecaseLetter => '\u01C5',
        UnicodeCategory.ModifierLetter => '\u02B0',
        UnicodeCategory.OtherLetter => '\u4E2D',
        UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or
            UnicodeCategory.EnclosingMark => '\u0301',
        UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or
            UnicodeCategory.OtherNumber => '0',
        _ => '\uE000',
    };
}
