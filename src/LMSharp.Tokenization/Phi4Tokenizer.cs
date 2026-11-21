using System.Buffers;
using System.Security.Cryptography;
using System.Text;
using LMSharp.Gguf;

namespace LMSharp.Tokenization;

public sealed class Phi4Tokenizer
{
    private const string ChatTemplateHash = "FEBF589225C9728AB791F52E8897D7607A823D45368F0A4C92FA68997B40CCE9";

    private static readonly (string Text, bool Special, bool TrimAfter)[] AddedTokenRules =
    [
        ("<|endoftext|>", true, false),
        ("<|endofprompt|>", true, false),
        ("<|assistant|>", true, true),
        ("<|end|>", true, true),
        ("<|user|>", true, true),
        ("<|system|>", true, true),
        ("<|tool|>", false, true),
        ("<|/tool|>", false, true),
        ("<|tool_call|>", false, true),
        ("<|/tool_call|>", false, true),
        ("<|tool_response|>", false, true),
        ("<|tag|>", true, true),
    ];

    private readonly string[] tokens;
    private readonly int[] types;
    private readonly Dictionary<int, AddedToken> addedById;
    private readonly AddedToken[] addedTokens;
    private readonly BytePairEncoder encoder;

    public Phi4Tokenizer(GgufFile model)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.Architecture != "phi3" ||
            model.RequireMetadata<string>("tokenizer.ggml.model") != "gpt2" ||
            model.RequireMetadata<string>("tokenizer.ggml.pre") != "gpt-4o" ||
            model.RequireMetadata<bool>("tokenizer.ggml.add_bos_token") ||
            model.RequireMetadata<bool>("tokenizer.ggml.add_eos_token"))
        {
            throw new GgufFormatException("Expected the Phi-4 Mini gpt-4o byte-level BPE profile.");
        }

        var template = model.RequireMetadata<string>("tokenizer.chat_template");
        if (Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(template))) != ChatTemplateHash)
        {
            throw new GgufFormatException("Unsupported Phi-4 Mini chat template.");
        }

        var tokenMetadata = model.RequireMetadata<IReadOnlyList<object>>("tokenizer.ggml.tokens");
        var typeMetadata = model.RequireMetadata<IReadOnlyList<object>>("tokenizer.ggml.token_type");
        if (tokenMetadata.Count != typeMetadata.Count)
        {
            throw new GgufFormatException("GGUF tokenizer tokens and token types differ in count.");
        }

        tokens = new string[tokenMetadata.Count];
        types = new int[tokenMetadata.Count];
        var vocabulary = new Dictionary<string, int>(tokenMetadata.Count, StringComparer.Ordinal);
        for (var id = 0; id < tokens.Length; id++)
        {
            if (tokenMetadata[id] is not string token || typeMetadata[id] is not (1 or 3 or 5) ||
                !vocabulary.TryAdd(token, id))
            {
                throw new GgufFormatException($"Invalid Phi-4 Mini vocabulary entry at token ID {id}.");
            }

            tokens[id] = token;
            types[id] = (int)typeMetadata[id];
        }

        addedTokens = AddedTokenRules.Select(rule =>
        {
            if (!vocabulary.TryGetValue(rule.Text, out var id) || types[id] != 3)
            {
                throw new GgufFormatException($"Missing Phi-4 Mini added token '{rule.Text}'.");
            }

            return new AddedToken(id, rule.Text, rule.Special, rule.TrimAfter);
        }).ToArray();
        addedById = addedTokens.ToDictionary(token => token.Id);

        EndOfTextId = vocabulary["<|endoftext|>"];
        EndOfTurnId = vocabulary["<|end|>"];
        if (model.RequireMetadata<uint>("tokenizer.ggml.eos_token_id") != EndOfTextId)
        {
            throw new GgufFormatException("Unexpected Phi-4 Mini end-of-text ID.");
        }

        encoder = new BytePairEncoder(vocabulary, model.RequireMetadata<IReadOnlyList<object>>("tokenizer.ggml.merges"));
    }

    public int VocabularySize => tokens.Length;
    public int EndOfTextId { get; }
    public int EndOfTurnId { get; }

    public static string FormatPrompt(string user, string? system = null)
    {
        ArgumentNullException.ThrowIfNull(user);
        return (system is null ? "" : $"<|system|>{system}<|end|>") +
            $"<|user|>{user}<|end|><|assistant|>";
    }

    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var result = new List<int>();
        var pendingStart = 0;
        var candidate = text.IndexOf("<|", StringComparison.Ordinal);
        while (candidate >= 0)
        {
            var added = addedTokens.FirstOrDefault(token =>
                text.AsSpan(candidate).StartsWith(token.Text, StringComparison.Ordinal));
            if (added is null)
            {
                candidate = text.IndexOf("<|", candidate + 2, StringComparison.Ordinal);
                continue;
            }

            AppendPlainTokens(text[pendingStart..candidate], result);
            result.Add(added.Id);
            pendingStart = candidate + added.Text.Length;
            if (added.TrimAfter)
            {
                while (pendingStart < text.Length && char.IsWhiteSpace(text[pendingStart]))
                {
                    pendingStart++;
                }
            }

            candidate = text.IndexOf("<|", pendingStart, StringComparison.Ordinal);
        }

        AppendPlainTokens(text[pendingStart..], result);
        return result;
    }

    public string Decode(IEnumerable<int> ids, bool skipSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var bytes = new ArrayBufferWriter<byte>();
        foreach (var id in ids)
        {
            if ((uint)id >= (uint)tokens.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(ids), $"Token ID {id} is outside the vocabulary.");
            }

            if (types[id] == 1)
            {
                BytePairEncoder.AppendDecodedBytes(tokens[id], bytes);
            }
            else if (!skipSpecialTokens || (addedById.TryGetValue(id, out var added) && !added.Special))
            {
                var text = tokens[id];
                var count = Encoding.UTF8.GetBytes(text, bytes.GetSpan(Encoding.UTF8.GetMaxByteCount(text.Length)));
                bytes.Advance(count);
            }
        }

        return Encoding.UTF8.GetString(bytes.WrittenSpan);
    }

    private void AppendPlainTokens(string text, List<int> result)
    {
        foreach (var piece in Phi4Pretokenizer.Split(text))
        {
            encoder.AppendTokens(piece, result);
        }
    }

    private sealed record AddedToken(int Id, string Text, bool Special, bool TrimAfter);
}
