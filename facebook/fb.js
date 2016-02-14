var casper = require('casper').create({
    pageSettings: {
        loadImages: false,//The script is much faster when this field is set to false
        loadPlugins: false,
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36'
    }
});
 
//First step is to open Facebook
casper.start().thenOpen("https://www.facebook.com", function() {
    console.log("Facebook website opened");
//     this.capture('BeforeLogin.png');
});
 
 
//Now we have to populate username and password, and submit the form
casper.then(function(){
    console.log("Login using username and password");
    this.evaluate(function(){
        document.getElementById("email").value="emmcee.elleff";
		document.getElementById("pass").value="mclf-password";
		document.getElementById("loginbutton").children[0].click();
    });
});

var group = "https://www.facebook.com/groups/" + casper.cli.options.group + "/members/" 
console.log(group);

casper.thenOpen(group, function() {
	this.echo(this.getTitle());
});

casper.repeat(+casper.cli.options.n, function() {
	casper.wait(1000, function() {
		this.evaluate(function() {
			var more = document.querySelector('a[href*="/list/"]');
			more.click();
		});
	});
});

casper.wait(3000, function() {
	var images = this.evaluate(function(){
		var facebookImages = document.getElementsByTagName('img'); 
		var allSrc = [];
		for(var i = 0; i < facebookImages.length; i++) {
			if(facebookImages[i].height >= 100 && facebookImages[i].width >= 100 && 
				facebookImages[i].height == facebookImages[i].width) {
				allSrc.push(facebookImages[i].src);
			}
		}
		return allSrc;
	});
	for (var i in images) {
		console.log(i + "  " + images[i]);
		this.download(images[i], i + ".jpg");
	}
});
 
casper.run();