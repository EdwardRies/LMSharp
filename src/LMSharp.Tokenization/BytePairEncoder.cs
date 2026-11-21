using System.Buffers;
using System.Text;
using LMSharp.Gguf;

namespace LMSharp.Tokenization;

internal sealed class BytePairEncoder
{
    private static readonly char[] ByteSymbols = CreateByteSymbols();
    private static readonly Dictionary<char, byte> SymbolBytes = ByteSymbols
        .Select((symbol, value) => (symbol, value))
        .ToDictionary(entry => entry.symbol, entry => (byte)entry.value);

    private readonly int[] byteIds;
    private readonly Dictionary<(int Left, int Right), Merge> merges;

    public BytePairEncoder(IReadOnlyDictionary<string, int> vocabulary, IReadOnlyList<object> rules)
    {
        byteIds = ByteSymbols.Select(symbol => GetId(symbol.ToString())).ToArray();
        merges = new Dictionary<(int, int), Merge>(rules.Count);
        for (var rank = 0; rank < rules.Count; rank++)
        {
            if (rules[rank] is not string rule)
            {
                throw new GgufFormatException($"Invalid BPE merge at rank {rank}.");
            }

            var separator = rule.IndexOf(' ');
            if (separator <= 0 || separator == rule.Length - 1)
            {
                throw new GgufFormatException($"Invalid BPE merge at rank {rank}.");
            }

            var left = rule[..separator];
            var right = rule[(separator + 1)..];
            if (!merges.TryAdd((GetId(left), GetId(right)), new Merge(rank, GetId(left + right))))
            {
                throw new GgufFormatException($"Duplicate BPE merge at rank {rank}.");
            }
        }

        int GetId(string symbol) => vocabulary.TryGetValue(symbol, out var id)
            ? id
            : throw new GgufFormatException($"BPE symbol '{symbol}' is missing from the vocabulary.");
    }

    public void AppendTokens(string piece, List<int> result)
    {
        var bytes = Encoding.UTF8.GetBytes(piece);
        if (bytes.Length == 1)
        {
            result.Add(byteIds[bytes[0]]);
            return;
        }

        var symbols = new int[bytes.Length];
        var previous = new int[bytes.Length];
        var next = new int[bytes.Length];
        var queue = new PriorityQueue<Candidate, (int Rank, int Position)>();

        for (var i = 0; i < bytes.Length; i++)
        {
            symbols[i] = byteIds[bytes[i]];
            previous[i] = i - 1;
            next[i] = i + 1;
        }

        for (var i = 0; i < bytes.Length - 1; i++)
        {
            Enqueue(i);
        }

        while (queue.TryDequeue(out var pair, out _))
        {
            if (next[pair.Left] != pair.Right || symbols[pair.Left] != pair.LeftId ||
                symbols[pair.Right] != pair.RightId)
            {
                continue;
            }

            symbols[pair.Left] = pair.MergedId;
            symbols[pair.Right] = -1;
            var after = next[pair.Right];
            next[pair.Left] = after;
            if (after < bytes.Length)
            {
                previous[after] = pair.Left;
            }

            Enqueue(previous[pair.Left]);
            Enqueue(pair.Left);
        }

        for (var i = 0; i < bytes.Length; i = next[i])
        {
            result.Add(symbols[i]);
        }

        void Enqueue(int left)
        {
            if (left >= 0 && next[left] < bytes.Length)
            {
                var right = next[left];
                var leftId = symbols[left];
                var rightId = symbols[right];
                if (merges.TryGetValue((leftId, rightId), out var merge))
                {
                    queue.Enqueue(new Candidate(left, right, leftId, rightId, merge.OutputId), (merge.Rank, left));
                }
            }
        }
    }

    public static void AppendDecodedBytes(string token, IBufferWriter<byte> output)
    {
        var bytes = output.GetSpan(token.Length);
        for (var i = 0; i < token.Length; i++)
        {
            if (!SymbolBytes.TryGetValue(token[i], out bytes[i]))
            {
                throw new GgufFormatException($"BPE token '{token}' contains an invalid byte symbol.");
            }
        }

        output.Advance(token.Length);
    }

    private static char[] CreateByteSymbols()
    {
        var symbols = new char[256];
        var escaped = 256;
        for (var value = 0; value < symbols.Length; value++)
        {
            symbols[value] = (char)(value is >= 33 and <= 126 or >= 161 and <= 172 or >= 174
                ? value
                : escaped++);
        }

        return symbols;
    }

    private readonly record struct Merge(int Rank, int OutputId);
    private readonly record struct Candidate(int Left, int Right, int LeftId, int RightId, int MergedId);
}
