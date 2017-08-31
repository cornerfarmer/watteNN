jQuery(document).ready(function () {
    initialize();
});

var cards = [];
var cards_left = [];
var players = [{hand_cards: [], board_card: null, tricks: 0}, {hand_cards: [], board_card: null, tricks: 0}];
var number_of_hand_cards = 5;
var current_player = 0;
var colors = {
    EICHEL: 1,
    GRUEN: 2,
    HERZ: 3,
    SCHELLN: 4
};

var values = {
    SAU: 8,
    KOENIG: 7,
    OBER: 6,
    UNTER: 5,
    ZEHN: 4,
    NEUN: 3,
    ACHT: 2,
    SIEBEN: 1
};

function initialize() {
    jQuery.get('model.json', function (data) {

        net = new convnetjs.Net(); // create an empty network
        net.fromJSON(data);

        var x_arr = new Array(68);

        for (var i = 0; i < x_arr.length; i++)
            x_arr[i] = 1;

        var x = new convnetjs.Vol(x_arr);

        var probability_volume = net.forward(x);
        console.log('probability that x is class 0: ' + probability_volume.w);
        console.log(net.toJSON());

        for (var color in colors) {
            for (var value in values) {
                cards.push({color: colors[color], value: values[value]})
            }
        }

        reset();
    });
}

function shuffle(a) {
    var j, x, i;
    for (i = a.length; i; i--) {
        j = Math.floor(Math.random() * i);
        x = a[i - 1];
        a[i - 1] = a[j];
        a[j] = x;
    }
}

function reset() {
    cards_left = cards.slice(0);
    shuffle(cards_left);
    for (var i = 0; i < players.length; i++) {
        players[i].hand_cards = [];
        players[i].board_card = null;
        players[i].tricks = 0;
        for (var c = 0; c < number_of_hand_cards; c++)
            players[i].hand_cards.push(cards_left.pop());
    }
    current_player = 1;
    refreshState("");
    refreshView();
}

function createCardHTML(card, number_in_hand_cards) {
    return '<div class="card" data-card-nr="' + number_in_hand_cards + '" style="background-image: url(\'../cards/' + filenameFromCard(card) + '.png\')"></div>'
}

function filenameFromCard(card) {
    var filename = "";
    if (card.color === colors.EICHEL)
        filename += "E";
    else if (card.color === colors.GRUEN)
        filename += "G";
    else if (card.color === colors.HERZ)
        filename += "H";
    else if (card.color === colors.SCHELLN)
        filename += "S";

    if (card.value === values.SAU)
        filename += "A";
    else if (card.value === values.KOENIG)
        filename += "K";
    else if (card.value === values.OBER)
        filename += "O";
    else if (card.value === values.UNTER)
        filename += "U";
    else if (card.value === values.ZEHN)
        filename += "10";
    else if (card.value === values.NEUN)
        filename += "9";
    else if (card.value === values.ACHT)
        filename += "8";
    else if (card.value === values.SIEBEN)
        filename += "7";
    return filename
}

function set_card(card) {
    var index = players[current_player].hand_cards.indexOf(card);
    var action = "";
    if (index >= 0) {
        if (players[0].board_card !== null && players[1].board_card !== null) {
            players[0].board_card = null;
            players[1].board_card = null;
        }

        players[current_player].board_card = players[current_player].hand_cards.splice(index, 1)[0];

        if (players[0].board_card !== null && players[1].board_card !== null) {
            var best_player = match(players[1 - current_player].board_card, players[current_player].board_card);

            if (best_player === 0)
                current_player = 1 - current_player;

            players[self.current_player].tricks += 1
        }
        else
            current_player = 1 - current_player;

        if (players[0].tricks === 3)
        {
            action = "AI won!";
            setTimeout(reset, 1000);
        }
        else if (players[1].tricks === 3)
        {
            action = "Human won!";
            setTimeout(reset, 1000);
        }
        else if (players[1].hand_cards.length + players[0].hand_cards.length === 0)
            setTimeout(reset, 1000);
        else if (current_player === 0)
            setTimeout(runAI, 1000);
    }
    else
    {
        action = "AI made an invalid move.";
        setTimeout(reset, 1000);
    }
    refreshState(action);
    refreshView();
}


function refreshView() {
    $("#player-ai .cards").empty();
    for (var i = 0; i < players[0].hand_cards.length; i++) {
        $("#player-ai .cards").append(createCardHTML(players[0].hand_cards[i], i));
    }
    $("#boardcard-ai").empty();
    if (players[0].board_card !== null)
        $("#boardcard-ai").append(createCardHTML(players[0].board_card, i));

    $("#player-human .cards").empty();
    for (var i = 0; i < players[1].hand_cards.length; i++) {
        $("#player-human .cards").append(createCardHTML(players[1].hand_cards[i], i));
    }
    $("#boardcard-human").empty();
    if (players[1].board_card !== null)
        $("#boardcard-human").append(createCardHTML(players[1].board_card, i));

    $("#player-human .cards .card").click(function () {
        if (current_player === 1) {
            set_card(players[1].hand_cards[$(this).attr('data-card-nr')]);
        }
    })

}

function addTrickArray(arr, player)
{
    if (player.tricks === 0) {
        arr.push(0);
        arr.push(0);
    }
    else if (player.tricks === 1) {
        arr.push(1);
        arr.push(0);
    }
    else if (player.tricks === 2) {
        arr.push(0);
        arr.push(1);
    }
    else if (player.tricks === 3) {
        arr.push(1);
        arr.push(1);
    }
}

function indexOfMax(arr)
{
    var max = arr[0];
    var maxIndex = 0;
    for (var i = 0; i < arr.length; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}


function runAI()
{
    obs = [];
    for (var i = 0; i < cards.length; i++) {
        obs.push(players[0].hand_cards.indexOf(cards[i]) !== -1 ? 1 : 0);
    }
    for (var i = 0; i < cards.length; i++) {
        obs.push(cards[i] === players[1].board_card ? 1 : 0);
    }

    addTrickArray(obs, players[0]);
    addTrickArray(obs, players[1]);

    var output = net.forward(new convnetjs.Vol(obs));
    var index = indexOfMax(output.w);
    set_card(cards[index]);
}

function match(first_card, second_card)
{
    if (getCardValue(first_card, first_card) >= getCardValue(second_card, first_card))
        return 0;
    else
        return 1;
}

function getCardValue(card, first_card)
{
    if (card.color === colors.HERZ && card.value === values.KOENIG)
        return 11;
    else if (card.color === colors.SCHELLN && card.value === values.SIEBEN)
        return 10;
    else if (card.color === colors.EICHEL && card.value === values.SIEBEN)
        return 9;

    if (card.color === first_card.color)
        return card.value;
    else
        return 0;
}

function refreshState(lastAction)
{
    $("#state").html((current_player === 1 ? "It's your turn" : "It's AI turn") + (lastAction !== "" ? " - " + lastAction : ""));
}