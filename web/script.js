jQuery(document).ready(function () {
    initialize();
});

var cards = [];
var cards_left = [];
var players = [{hand_cards: [], board_card: null, tricks: 0}, {hand_cards: [], board_card: null, tricks: 0}];
var number_of_hand_cards = 3;
var current_player = 0;
var next_start_player = 1;
var colors = {
    EICHEL: 0,
    GRUEN: 1,
    HERZ: 2,
    SCHELLN: 3
};

var values = {
    SAU: 7,
    KOENIG: 6,
    OBER: 5,
    UNTER: 4,
    ZEHN: 3,
    NEUN: 2,
    ACHT: 1,
    SIEBEN: 0
};
var model;

function initialize() {
    model = new KerasJS.Model({
      filepath: 'modelDense3.bin'
    });

    for (var color = 0; color < 2; color++) {
        for (var value = 7; value >= 4; value--) {
            //cards.push({color: colors[color], value: values[value]})
            cards.push({color: color, value: value});
        }
    }

    reset();
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
    current_player = next_start_player;
    next_start_player = 1 - next_start_player;
    refreshState("");
    refreshView();

    if (current_player === 0)
        setTimeout(runAI, 1000);
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

        if (players[0].tricks === 2)
        {
            action = "AI won!";
            setTimeout(reset, 1000);
        }
        else if (players[1].tricks === 2)
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
    obs_cube = [];
    for (var c = 0; c < 4 * 8 * 2; c++) {
        obs_cube.push(0);
    }
    for (var i = 0; i < players[0].hand_cards.length; i++) {
        obs_cube[players[0].hand_cards[i].color * 8 * 2 + players[0].hand_cards[i].value * 2 + 0] = 1;
    }
    if (players[1].board_card !== null)
        obs_cube[players[1].board_card.color * 8 * 2 + players[1].board_card.value * 2 + 1] = 1;


    obs_flat = [];
    addTrickArray(obs_flat, players[0]);
    addTrickArray(obs_flat, players[1]);

    model.ready()
    .then(() => {
        const inputData = {
          input_1: new Float32Array(obs_cube),
          input_2: new Float32Array(obs_flat)
        }

        // make predictions
        return model.predict(inputData)
    }).then(outputData => {
        console.log(outputData);
        var index = indexOfMax(outputData['dense_2']);
        set_card(cards[index]);
    })

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